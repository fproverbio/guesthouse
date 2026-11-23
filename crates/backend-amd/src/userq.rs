//! `AMDGPU_USERQ`: creating, feeding and freeing a kernel-tracked user queue.
//!
//! A user queue's residency is the kernel's eviction-fence mechanism, not a per-submission buffer
//! list: once its buffers are mapped into this file's VM, the daemon writes command packets into
//! the queue's own ring directly and rings a doorbell, with no `DRM_AMDGPU_CS` ioctl in the steady
//! state. Completion is `DRM_AMDGPU_USERQ_SIGNAL`, read against the ring's own write pointer: the
//! kernel samples the wptr buffer we've just written and tells the named sync objects apart once
//! the queue's hardware reaches it, the same signal a submission's eventfd already waits on.
//!
//! The sequencing here (ring, then a store fence, then the wptr buffer, then another store fence,
//! then the doorbell) matches radeonsi's own userqueue winsys
//! (`src/gallium/winsys/amdgpu/drm/amdgpu_userq.c`/`amdgpu_cs.cpp` in Mesa), which wires up GFX and
//! COMPUTE and is the only production user of this uAPI — but every queue shape here, GFX and
//! COMPUTE included, is derived straight from kernel source, not ported from Mesa's own code
//! (unavailable to fetch while this was written). [`SdmaUserq`] has no Mesa reference at all: it
//! and [`GfxUserq`]/[`ComputeUserq`] also differ in [`UserQueue::submit`]'s wptr/doorbell
//! convention — DMA's is byte-granular, the other two's dword-granular (`sdma_v6_0_ring_set_wptr`'s
//! `<< 2` against `gfx_v12_0_ring_set_wptr_gfx`/`_compute`'s raw write).

use crate::device::Mapping;
use crate::info::{HW_IP_COMPUTE, HW_IP_DMA, HW_IP_GFX};
use crate::ioctl::{self, call, get32, put32, put64};
use crate::memory::GemHandle;
use crate::submit::Syncobj;
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{Ordering, fence};

const OP_CREATE: u32 = 1;
const OP_FREE: u32 = 2;

// Offsets into drm_amdgpu_userq_in / _out, the 72-byte request the union is sized for.
const IN_OP: usize = 0;
const IN_QUEUE_ID: usize = 4;
const IN_IP_TYPE: usize = 8;
const IN_DOORBELL_HANDLE: usize = 12;
const IN_DOORBELL_OFFSET: usize = 16;
const IN_QUEUE_VA: usize = 24;
const IN_QUEUE_SIZE: usize = 32;
const IN_RPTR_VA: usize = 40;
const IN_WPTR_VA: usize = 48;
const IN_MQD: usize = 56;
const IN_MQD_SIZE: usize = 64;
const OUT_QUEUE_ID: usize = 0;

// Offsets into drm_amdgpu_userq_signal, 56 bytes.
const SIGNAL_QUEUE_ID: usize = 0;
const SIGNAL_SYNCOBJ_HANDLES: usize = 8;
const SIGNAL_NUM_SYNCOBJ_HANDLES: usize = 16;
const SIGNAL_REQUEST_SIZE: usize = 56;

/// An offset into the doorbell buffer, as a count of 8-byte slots: both the ioctl's
/// `doorbell_offset` and the slot this writes the new write pointer to are this queue's own
/// private page, so any value works, and 0 keeps the one page this backend gives each queue to a
/// single slot. See `AMDGPU_USERQ_DOORBELL_INDEX` in Mesa's winsys for the same convention.
const DOORBELL_SLOT: usize = 0;

/// Whether this queue's wptr/doorbell value is the ring position's raw dword count, or that
/// shifted into bytes. GFX and COMPUTE use dwords (`gfx_v11_0_ring_set_wptr_compute`'s
/// `WDOORBELL64(ring->doorbell_index, ring->wptr)` writes it unshifted); DMA uses bytes
/// (`sdma_v6_0_ring_set_wptr`'s `ring->wptr << 2`). See this module's own doc comment.
#[derive(Clone, Copy, Debug)]
enum WptrUnit {
    Dwords,
    Bytes,
}

/// `drm_amdgpu_userq_mqd_sdma_gfx11`: the DMA engine's only per-queue metadata, naming a
/// context-save area by its GPU virtual address. The kernel reads this by a host pointer in the
/// ioctl request, not as a GPU address itself.
#[repr(C)]
struct SdmaMqd {
    csa_va: u64,
}

/// `drm_amdgpu_userq_mqd_compute_gfx11`: the COMPUTE engine's only per-queue metadata, naming an
/// end-of-pipe buffer sized and aligned by [`crate::info::query_userq_compute_eop`].
#[repr(C)]
struct ComputeMqd {
    eop_va: u64,
}

/// `drm_amdgpu_userq_mqd_gfx11`: the GFX engine's per-queue metadata, naming a shadow buffer and a
/// context-save area, both sized and aligned by [`crate::info::query_userq_gfx_shadow_csa`].
#[repr(C)]
struct GfxMqd {
    shadow_va: u64,
    csa_va: u64,
}

/// The engine-agnostic half of a kernel-tracked user queue: a ring this backend writes packets
/// into directly, instead of a `DRM_AMDGPU_CS` ioctl per submission. [`SdmaUserq`] and
/// [`ComputeUserq`] are thin, engine-specific wrappers around this, differing only in which
/// `ip_type`/MQD the kernel is told and which [`WptrUnit`] its hardware expects.
#[derive(Debug)]
struct UserQueue {
    id: u32,
    ring: Mapping,
    /// The ring's size in dwords, which must be a power of two: the ring position is the write
    /// pointer modulo this by a mask, as amdgpu's own MES-scheduled queues read it.
    ring_size_dw: u64,
    wptr: Mapping,
    doorbell: Mapping,
    /// The ring position past the last dword written, monotonically increasing and never
    /// wrapped itself, as the uAPI and the kernel's own progress tracking both expect.
    next_wptr: u64,
    wptr_unit: WptrUnit,
}

impl UserQueue {
    /// `ring`/`ring_va`/`ring_size` names the ring buffer, already mapped in this file's VM at
    /// `ring_va`; `ring_size` must be a power of two, in bytes. `rptr_va` is a buffer of at least
    /// 8 bytes, likewise mapped, that this backend never reads: the kernel tracks the hardware's
    /// own progress through it. `wptr`/`wptr_va` is where this queue's write pointer is told to
    /// the kernel, and `doorbell`/`doorbell_va`'s first 8-byte slot is where ringing it is told to
    /// the card; both need a CPU mapping as well as the GPU one the caller has already made.
    /// `mqd`/`mqd_size` is the engine-specific metadata, already laid out as the kernel's uAPI
    /// struct for `ip_type` expects, and must outlive this call (it's read by a host pointer in
    /// the ioctl request, not copied in beforehand).
    #[allow(clippy::too_many_arguments)]
    fn create(
        file: &File,
        ip_type: u32,
        ring: GemHandle,
        ring_va: u64,
        ring_size: u64,
        rptr_va: u64,
        wptr: GemHandle,
        wptr_va: u64,
        doorbell: GemHandle,
        mqd: u64,
        mqd_size: u64,
        wptr_unit: WptrUnit,
    ) -> io::Result<Self> {
        assert!(ring_size.is_power_of_two(), "a user queue's ring is a power of two in size");
        let ring_map = Mapping::of(file, ring, ring_size as usize)?;
        let wptr_map = Mapping::of(file, wptr, 8)?;
        let doorbell_map = Mapping::of(file, doorbell, (DOORBELL_SLOT + 1) * 8)?;

        let mut request = [0u8; 72];
        put32(&mut request, IN_OP, OP_CREATE);
        put32(&mut request, IN_IP_TYPE, ip_type);
        put32(&mut request, IN_DOORBELL_HANDLE, doorbell.0);
        put32(&mut request, IN_DOORBELL_OFFSET, DOORBELL_SLOT as u32);
        put64(&mut request, IN_QUEUE_VA, ring_va);
        put64(&mut request, IN_QUEUE_SIZE, ring_size);
        put64(&mut request, IN_RPTR_VA, rptr_va);
        put64(&mut request, IN_WPTR_VA, wptr_va);
        put64(&mut request, IN_MQD, mqd);
        put64(&mut request, IN_MQD_SIZE, mqd_size);
        ioctl::drm_ioctl(file, call::userq(), &mut request)?;
        Ok(Self {
            id: get32(&request, OUT_QUEUE_ID),
            ring: ring_map,
            ring_size_dw: ring_size / 4,
            wptr: wptr_map,
            doorbell: doorbell_map,
            next_wptr: 0,
            wptr_unit,
        })
    }

    /// Writes `dwords` to the ring and rings the doorbell, without waiting. The kernel signals
    /// `signal` once the queue's hardware reaches the write pointer this leaves it at, the same
    /// signal an eventfd armed on it already waits on.
    fn submit(&mut self, file: &impl AsRawFd, dwords: &[u32], signal: &Syncobj) -> io::Result<()> {
        let mask = self.ring_size_dw - 1;
        for (offset, &word) in dwords.iter().enumerate() {
            let position = (self.next_wptr + offset as u64) & mask;
            self.ring.write_u32(position as usize * 4, word);
        }
        self.next_wptr += dwords.len() as u64;

        let wptr_value = match self.wptr_unit {
            WptrUnit::Dwords => self.next_wptr,
            WptrUnit::Bytes => self.next_wptr * 4,
        };

        // The ring's new dwords must reach memory before the wptr update that tells the kernel
        // they're there, which must itself reach memory before the doorbell write that has the
        // card act on it: both are reads the hardware does on its own, not ones this process's
        // instruction order alone would order correctly against a weakly-ordered mapping.
        fence(Ordering::SeqCst);
        self.wptr.write_u32(0, wptr_value as u32);
        self.wptr.write_u32(4, (wptr_value >> 32) as u32);
        fence(Ordering::SeqCst);
        self.doorbell.write_u32(DOORBELL_SLOT * 8, wptr_value as u32);
        self.doorbell.write_u32(DOORBELL_SLOT * 8 + 4, (wptr_value >> 32) as u32);

        let handles = [signal.handle()];
        let mut request = [0u8; SIGNAL_REQUEST_SIZE];
        put32(&mut request, SIGNAL_QUEUE_ID, self.id);
        put64(&mut request, SIGNAL_SYNCOBJ_HANDLES, handles.as_ptr() as u64);
        ioctl::put16(&mut request, SIGNAL_NUM_SYNCOBJ_HANDLES, handles.len() as u16);
        ioctl::drm_ioctl(file, call::userq_signal(), &mut request)
    }

    /// Frees the queue. Buffers it named, and `signal`s from [`Self::submit`], are the caller's
    /// own and outlive it.
    fn destroy(&self, file: &impl AsRawFd) {
        let mut request = [0u8; 72];
        put32(&mut request, IN_OP, OP_FREE);
        put32(&mut request, IN_QUEUE_ID, self.id);
        let _ = ioctl::drm_ioctl(file, call::userq(), &mut request);
    }
}

/// A kernel-tracked user queue on the DMA engine: a ring this backend writes SDMA packets into
/// directly, instead of a `DRM_AMDGPU_CS` ioctl per submission.
#[derive(Debug)]
pub struct SdmaUserq(UserQueue);

impl SdmaUserq {
    /// See [`UserQueue::create`]. `csa_va` is a context-save area sized and aligned by
    /// [`crate::info::query_userq_sdma_csa`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        file: &File,
        ring: GemHandle,
        ring_va: u64,
        ring_size: u64,
        rptr_va: u64,
        wptr: GemHandle,
        wptr_va: u64,
        doorbell: GemHandle,
        csa_va: u64,
    ) -> io::Result<Self> {
        let mqd = SdmaMqd { csa_va };
        UserQueue::create(
            file,
            HW_IP_DMA,
            ring,
            ring_va,
            ring_size,
            rptr_va,
            wptr,
            wptr_va,
            doorbell,
            std::ptr::from_ref(&mqd) as u64,
            std::mem::size_of::<SdmaMqd>() as u64,
            WptrUnit::Bytes,
        )
        .map(Self)
    }

    /// See [`UserQueue::submit`].
    pub fn submit(&mut self, file: &impl AsRawFd, dwords: &[u32], signal: &Syncobj) -> io::Result<()> {
        self.0.submit(file, dwords, signal)
    }

    /// See [`UserQueue::destroy`].
    pub fn destroy(&self, file: &impl AsRawFd) {
        self.0.destroy(file);
    }
}

/// A kernel-tracked user queue on the GFX engine: a ring this backend writes PM4 packets into
/// directly, instead of a `DRM_AMDGPU_CS` ioctl per submission. MES places it on a hardware queue
/// slot the kernel's own graphics ring never claims (queue 0 of each pipe is the kernel's own
/// KGQ; MES keeps user queues to queue 1 and up — see `amdgpu_mes_get_hqd_mask`), so unlike the
/// collision this crate's own COMPUTE path can't yet rule out, GFX's completion interrupt is
/// never at risk of being claimed by a kernel-owned ring instead of reaching this queue's fence.
#[derive(Debug)]
pub struct GfxUserq(UserQueue);

impl GfxUserq {
    /// See [`UserQueue::create`]. `shadow_va` and `csa_va` are sized and aligned by
    /// [`crate::info::query_userq_gfx_shadow_csa`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        file: &File,
        ring: GemHandle,
        ring_va: u64,
        ring_size: u64,
        rptr_va: u64,
        wptr: GemHandle,
        wptr_va: u64,
        doorbell: GemHandle,
        shadow_va: u64,
        csa_va: u64,
    ) -> io::Result<Self> {
        let mqd = GfxMqd { shadow_va, csa_va };
        UserQueue::create(
            file,
            HW_IP_GFX,
            ring,
            ring_va,
            ring_size,
            rptr_va,
            wptr,
            wptr_va,
            doorbell,
            std::ptr::from_ref(&mqd) as u64,
            std::mem::size_of::<GfxMqd>() as u64,
            WptrUnit::Dwords,
        )
        .map(Self)
    }

    /// See [`UserQueue::submit`].
    pub fn submit(&mut self, file: &impl AsRawFd, dwords: &[u32], signal: &Syncobj) -> io::Result<()> {
        self.0.submit(file, dwords, signal)
    }

    /// See [`UserQueue::destroy`].
    pub fn destroy(&self, file: &impl AsRawFd) {
        self.0.destroy(file);
    }
}

/// A kernel-tracked user queue on the COMPUTE engine: a ring this backend writes PM4 packets into
/// directly, instead of a `DRM_AMDGPU_CS` ioctl per submission.
#[derive(Debug)]
pub struct ComputeUserq(UserQueue);

impl ComputeUserq {
    /// See [`UserQueue::create`]. `eop_va` is an end-of-pipe buffer sized and aligned by
    /// [`crate::info::query_userq_compute_eop`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        file: &File,
        ring: GemHandle,
        ring_va: u64,
        ring_size: u64,
        rptr_va: u64,
        wptr: GemHandle,
        wptr_va: u64,
        doorbell: GemHandle,
        eop_va: u64,
    ) -> io::Result<Self> {
        let mqd = ComputeMqd { eop_va };
        UserQueue::create(
            file,
            HW_IP_COMPUTE,
            ring,
            ring_va,
            ring_size,
            rptr_va,
            wptr,
            wptr_va,
            doorbell,
            std::ptr::from_ref(&mqd) as u64,
            std::mem::size_of::<ComputeMqd>() as u64,
            WptrUnit::Dwords,
        )
        .map(Self)
    }

    /// See [`UserQueue::submit`].
    pub fn submit(&mut self, file: &impl AsRawFd, dwords: &[u32], signal: &Syncobj) -> io::Result<()> {
        self.0.submit(file, dwords, signal)
    }

    /// See [`UserQueue::destroy`].
    pub fn destroy(&self, file: &impl AsRawFd) {
        self.0.destroy(file);
    }
}
