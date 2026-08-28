//! Shared allocation instrumentation for benchmark binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingAllocator;

static BYTES_ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static ALLOC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        BYTES_ALLOCATED.fetch_add(layout.size(), Ordering::Relaxed);
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        let current = CURRENT_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
        PEAK_BYTES.fetch_max(current, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Run `f` once and return allocated bytes, allocation calls, and peak live
/// bytes attributable to that call.
pub(crate) fn count_allocations<T>(f: impl FnOnce() -> T) -> (usize, usize, usize) {
    let bytes_before = BYTES_ALLOCATED.load(Ordering::Relaxed);
    let calls_before = ALLOC_CALLS.load(Ordering::Relaxed);
    let current_before = CURRENT_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(current_before, Ordering::Relaxed);
    black_box(f());
    let peak_during = PEAK_BYTES.load(Ordering::Relaxed);
    (
        BYTES_ALLOCATED.load(Ordering::Relaxed) - bytes_before,
        ALLOC_CALLS.load(Ordering::Relaxed) - calls_before,
        peak_during.saturating_sub(current_before),
    )
}
