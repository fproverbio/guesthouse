//! Writing through a kernel-tracked user queue, on a real card whose kernel and firmware support
//! it for that engine: a command ring this backend feeds directly, with no `DRM_AMDGPU_CS` ioctl
//! per submission. Without an AMD render node, or one whose kernel doesn't report that engine's
//! `userq_ip_mask` bit, the matching test returns early.
//!
//! Different hosts can support different, disjoint engines here (this backend's own RDNA3 card
//! only ever reports the DMA bit; an RDNA4 tester's reported GFX and COMPUTE instead, never DMA),
//! so each test's own capability check already keeps it from running where the kernel can't back
//! it. `GUESTHOUSE_USERQ_SKIP` is a second, explicit gate on top of that: a comma-separated list
//! of engine names (`dma`, `compute`, `gfx`) to keep from running even where the kernel does
//! report support — for a tester who wants one left out of a run regardless, or a re-run of
//! everything but a path that just failed.

#![cfg(target_os = "linux")]

use guesthouse_backend_amd::{
    ComputeUserq, CpuAccess, DOMAIN_DOORBELL, DOMAIN_GTT, DOMAIN_VRAM, Device, GfxUserq, HW_IP_COMPUTE,
    HW_IP_DMA, HW_IP_GFX, MapFlags, MemoryType, SdmaUserq, Syncobj, close_buffer, create_buffer, map_buffer,
    pm4_release_mem_ib, pm4_write_ib, query_userq_compute_eop, query_userq_gfx_shadow_csa,
    query_userq_sdma_csa, sdma_fence_ib, sdma_trap_ib, sdma_write_ib, unmap_buffer, wait_eventfd,
};

const PAGE: u64 = 4096;
const MAGIC: u32 = 0x75E8_90D0;
const FENCE_MAGIC: u32 = 0xFE4C_E000;
const WAIT_MS: i32 = 2000;

/// See `GUESTHOUSE_USERQ_SKIP` in this file's own doc comment.
fn skip_requested(engine: &str) -> bool {
    std::env::var("GUESTHOUSE_USERQ_SKIP")
        .ok()
        .is_some_and(|list| list.split(',').any(|name| name.trim().eq_ignore_ascii_case(engine)))
}

#[test]
fn a_dma_user_queue_writes_through_a_card_whose_kernel_supports_it() {
    if skip_requested("dma") {
        eprintln!("GUESTHOUSE_USERQ_SKIP names dma; skipping the user queue proof");
        return;
    }
    let Ok(device) = Device::open(None) else {
        eprintln!("no AMD render node; skipping the user queue proof");
        return;
    };
    let info = device.info().clone();
    if info.userq_ip_mask & (1 << HW_IP_DMA) == 0 {
        eprintln!("this card's kernel offers no DMA user queue; skipping the user queue proof");
        return;
    }

    let align = info.va_alignment.max(PAGE);
    let flags =
        MapFlags { readable: true, writable: true, executable: true, memory_type: MemoryType::Uncached };
    let make = |bytes| create_buffer(&device, bytes, align, DOMAIN_GTT, CpuAccess::NotNeeded).unwrap();
    let (csa_size, csa_alignment) = query_userq_sdma_csa(&device).unwrap();
    assert_ne!(csa_size, 0, "a card offering DMA user queues names a context-save area size");

    let csa_bytes = u64::from(csa_size).max(1).next_multiple_of(align);
    let ring = make(PAGE);
    // The kernel and hardware write these two; radeonsi's own userqueue winsys puts them in VRAM.
    let rptr = create_buffer(&device, PAGE, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let wptr = make(PAGE);
    let csa = create_buffer(&device, csa_bytes, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let fence_target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let doorbell = create_buffer(
        &device,
        PAGE,
        u64::from(csa_alignment).max(align),
        DOMAIN_DOORBELL,
        CpuAccess::NotNeeded,
    )
    .unwrap();

    let base = info.va_start + 0x8_0000_0000;
    let (ring_va, rptr_va, wptr_va, csa_va, target_va, fence_target_va) = (
        base,
        base + PAGE,
        base + 2 * PAGE,
        base + 3 * PAGE,
        base + 4 * PAGE,
        base + 5 * PAGE,
    );
    map_buffer(&device, ring, ring_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, rptr, rptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, wptr, wptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, csa, csa_va, 0, csa_bytes, flags).unwrap();
    map_buffer(&device, target, target_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, fence_target, fence_target_va, 0, PAGE, flags).unwrap();
    let target_cpu = device.map_cpu(target, PAGE as usize).unwrap();
    let fence_target_cpu = device.map_cpu(fence_target, PAGE as usize).unwrap();

    let mut queue =
        SdmaUserq::create(device.file(), ring, ring_va, PAGE, rptr_va, wptr, wptr_va, doorbell, csa_va)
            .unwrap();

    let mut packet = sdma_write_ib(target_va, &[MAGIC]);
    packet.extend(sdma_fence_ib(fence_target_va, FENCE_MAGIC));
    packet.extend(sdma_trap_ib());
    eprintln!("packet: {packet:x?}");
    let signal = Syncobj::create(&device).unwrap();
    let submitted = queue.submit(&device, &packet, &signal);
    eprintln!("submit: {submitted:?}");
    let event = signal.arm_eventfd(&device).unwrap();
    let signalled = wait_eventfd(&event, WAIT_MS);
    eprintln!(
        "signalled: {signalled:?}, target word now {:#x}, fence target now {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    std::thread::sleep(std::time::Duration::from_millis(500));
    eprintln!(
        "after a pause, target word is {:#x}, fence target is {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    assert_eq!(target_cpu.read_u32(0), MAGIC, "the queue's write landed at the wrong place");
    signal.destroy(&device);

    queue.destroy(&device);
    unmap_buffer(&device, ring, ring_va, PAGE).unwrap();
    unmap_buffer(&device, rptr, rptr_va, PAGE).unwrap();
    unmap_buffer(&device, wptr, wptr_va, PAGE).unwrap();
    unmap_buffer(&device, csa, csa_va, csa_bytes).unwrap();
    unmap_buffer(&device, target, target_va, PAGE).unwrap();
    unmap_buffer(&device, fence_target, fence_target_va, PAGE).unwrap();
    for handle in [ring, rptr, wptr, csa, target, fence_target, doorbell] {
        close_buffer(&device, handle);
    }
}

#[test]
fn a_compute_user_queue_writes_through_a_card_whose_kernel_supports_it() {
    if skip_requested("compute") {
        eprintln!("GUESTHOUSE_USERQ_SKIP names compute; skipping the user queue proof");
        return;
    }
    let Ok(device) = Device::open(None) else {
        eprintln!("no AMD render node; skipping the user queue proof");
        return;
    };
    let info = device.info().clone();
    if info.userq_ip_mask & (1 << HW_IP_COMPUTE) == 0 {
        eprintln!("this card's kernel offers no COMPUTE user queue; skipping the user queue proof");
        return;
    }

    let align = info.va_alignment.max(PAGE);
    let flags =
        MapFlags { readable: true, writable: true, executable: true, memory_type: MemoryType::Uncached };
    let make = |bytes| create_buffer(&device, bytes, align, DOMAIN_GTT, CpuAccess::NotNeeded).unwrap();
    let (eop_size, eop_alignment) = query_userq_compute_eop(&device).unwrap();
    assert_ne!(eop_size, 0, "a card offering COMPUTE user queues names an end-of-pipe buffer size");

    let eop_bytes = u64::from(eop_size).max(1).next_multiple_of(align);
    let ring = make(PAGE);
    // The kernel and hardware write these two; radeonsi's own userqueue winsys puts them in VRAM.
    let rptr = create_buffer(&device, PAGE, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let wptr = make(PAGE);
    let eop = create_buffer(&device, eop_bytes, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let fence_target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let doorbell = create_buffer(
        &device,
        PAGE,
        u64::from(eop_alignment).max(align),
        DOMAIN_DOORBELL,
        CpuAccess::NotNeeded,
    )
    .unwrap();

    let base = info.va_start + 0x9_0000_0000;
    let (ring_va, rptr_va, wptr_va, eop_va, target_va, fence_target_va) = (
        base,
        base + PAGE,
        base + 2 * PAGE,
        base + 3 * PAGE,
        base + 4 * PAGE,
        base + 5 * PAGE,
    );
    map_buffer(&device, ring, ring_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, rptr, rptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, wptr, wptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, eop, eop_va, 0, eop_bytes, flags).unwrap();
    map_buffer(&device, target, target_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, fence_target, fence_target_va, 0, PAGE, flags).unwrap();
    let target_cpu = device.map_cpu(target, PAGE as usize).unwrap();
    let fence_target_cpu = device.map_cpu(fence_target, PAGE as usize).unwrap();

    let mut queue =
        ComputeUserq::create(device.file(), ring, ring_va, PAGE, rptr_va, wptr, wptr_va, doorbell, eop_va)
            .unwrap();

    let mut packet = pm4_write_ib(target_va, &[MAGIC]);
    packet.extend(pm4_release_mem_ib(fence_target_va, FENCE_MAGIC));
    eprintln!("packet: {packet:x?}");
    let signal = Syncobj::create(&device).unwrap();
    let submitted = queue.submit(&device, &packet, &signal);
    eprintln!("submit: {submitted:?}");
    let event = signal.arm_eventfd(&device).unwrap();
    let signalled = wait_eventfd(&event, WAIT_MS);
    eprintln!(
        "signalled: {signalled:?}, target word now {:#x}, fence target now {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    std::thread::sleep(std::time::Duration::from_millis(500));
    eprintln!(
        "after a pause, target word is {:#x}, fence target is {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    assert_eq!(target_cpu.read_u32(0), MAGIC, "the queue's write landed at the wrong place");
    signal.destroy(&device);

    queue.destroy(&device);
    unmap_buffer(&device, ring, ring_va, PAGE).unwrap();
    unmap_buffer(&device, rptr, rptr_va, PAGE).unwrap();
    unmap_buffer(&device, wptr, wptr_va, PAGE).unwrap();
    unmap_buffer(&device, eop, eop_va, eop_bytes).unwrap();
    unmap_buffer(&device, target, target_va, PAGE).unwrap();
    unmap_buffer(&device, fence_target, fence_target_va, PAGE).unwrap();
    for handle in [ring, rptr, wptr, eop, target, fence_target, doorbell] {
        close_buffer(&device, handle);
    }
}

#[test]
fn a_gfx_user_queue_writes_through_a_card_whose_kernel_supports_it() {
    if skip_requested("gfx") {
        eprintln!("GUESTHOUSE_USERQ_SKIP names gfx; skipping the user queue proof");
        return;
    }
    let Ok(device) = Device::open(None) else {
        eprintln!("no AMD render node; skipping the user queue proof");
        return;
    };
    let info = device.info().clone();
    if info.userq_ip_mask & (1 << HW_IP_GFX) == 0 {
        eprintln!("this card's kernel offers no GFX user queue; skipping the user queue proof");
        return;
    }

    let align = info.va_alignment.max(PAGE);
    let flags =
        MapFlags { readable: true, writable: true, executable: true, memory_type: MemoryType::Uncached };
    let make = |bytes| create_buffer(&device, bytes, align, DOMAIN_GTT, CpuAccess::NotNeeded).unwrap();
    let (shadow_size, shadow_alignment, csa_size, csa_alignment) = query_userq_gfx_shadow_csa(&device).unwrap();
    assert_ne!(shadow_size, 0, "a card offering GFX user queues names a shadow buffer size");
    assert_ne!(csa_size, 0, "a card offering GFX user queues names a context-save area size");

    let shadow_bytes = u64::from(shadow_size).max(1).next_multiple_of(align);
    let csa_bytes = u64::from(csa_size).max(1).next_multiple_of(align);
    let ring = make(PAGE);
    // The kernel and hardware write these two; radeonsi's own userqueue winsys puts them in VRAM.
    let rptr = create_buffer(&device, PAGE, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let wptr = make(PAGE);
    let shadow = create_buffer(&device, shadow_bytes, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let csa = create_buffer(&device, csa_bytes, align, DOMAIN_VRAM, CpuAccess::NotNeeded).unwrap();
    let target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let fence_target = create_buffer(&device, PAGE, align, DOMAIN_GTT, CpuAccess::Required).unwrap();
    let doorbell = create_buffer(
        &device,
        PAGE,
        u64::from(shadow_alignment).max(u64::from(csa_alignment)).max(align),
        DOMAIN_DOORBELL,
        CpuAccess::NotNeeded,
    )
    .unwrap();

    let base = info.va_start + 0xA_0000_0000;
    let (ring_va, rptr_va, wptr_va, shadow_va, csa_va, target_va, fence_target_va) = (
        base,
        base + PAGE,
        base + 2 * PAGE,
        base + 3 * PAGE,
        base + 3 * PAGE + shadow_bytes,
        base + 3 * PAGE + shadow_bytes + csa_bytes,
        base + 4 * PAGE + shadow_bytes + csa_bytes,
    );
    map_buffer(&device, ring, ring_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, rptr, rptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, wptr, wptr_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, shadow, shadow_va, 0, shadow_bytes, flags).unwrap();
    map_buffer(&device, csa, csa_va, 0, csa_bytes, flags).unwrap();
    map_buffer(&device, target, target_va, 0, PAGE, flags).unwrap();
    map_buffer(&device, fence_target, fence_target_va, 0, PAGE, flags).unwrap();
    let target_cpu = device.map_cpu(target, PAGE as usize).unwrap();
    let fence_target_cpu = device.map_cpu(fence_target, PAGE as usize).unwrap();

    let mut queue = GfxUserq::create(
        device.file(),
        ring,
        ring_va,
        PAGE,
        rptr_va,
        wptr,
        wptr_va,
        doorbell,
        shadow_va,
        csa_va,
    )
    .unwrap();

    let mut packet = pm4_write_ib(target_va, &[MAGIC]);
    packet.extend(pm4_release_mem_ib(fence_target_va, FENCE_MAGIC));
    eprintln!("packet: {packet:x?}");
    let signal = Syncobj::create(&device).unwrap();
    let submitted = queue.submit(&device, &packet, &signal);
    eprintln!("submit: {submitted:?}");
    let event = signal.arm_eventfd(&device).unwrap();
    let signalled = wait_eventfd(&event, WAIT_MS);
    eprintln!(
        "signalled: {signalled:?}, target word now {:#x}, fence target now {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    std::thread::sleep(std::time::Duration::from_millis(500));
    eprintln!(
        "after a pause, target word is {:#x}, fence target is {:#x}",
        target_cpu.read_u32(0),
        fence_target_cpu.read_u32(0)
    );
    assert_eq!(target_cpu.read_u32(0), MAGIC, "the queue's write landed at the wrong place");
    signal.destroy(&device);

    queue.destroy(&device);
    unmap_buffer(&device, ring, ring_va, PAGE).unwrap();
    unmap_buffer(&device, rptr, rptr_va, PAGE).unwrap();
    unmap_buffer(&device, wptr, wptr_va, PAGE).unwrap();
    unmap_buffer(&device, shadow, shadow_va, shadow_bytes).unwrap();
    unmap_buffer(&device, csa, csa_va, csa_bytes).unwrap();
    unmap_buffer(&device, target, target_va, PAGE).unwrap();
    unmap_buffer(&device, fence_target, fence_target_va, PAGE).unwrap();
    for handle in [ring, rptr, wptr, shadow, csa, target, fence_target, doorbell] {
        close_buffer(&device, handle);
    }
}
