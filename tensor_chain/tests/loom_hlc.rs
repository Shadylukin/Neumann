// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Loom-based concurrency verification tests for HLC.
//!
//! These tests exhaustively explore all thread interleavings to verify
//! that the HLC produces unique, monotonic timestamps under concurrency.
//!
//! Run with: cargo nextest run --package tensor_chain --features loom -E 'test(loom_)'

#![cfg(feature = "loom")]

use loom::sync::Arc;
use loom::thread;
use tensor_chain::hlc::HybridLogicalClock;

#[test]
fn loom_hlc_concurrent_now_unique() {
    loom::model(|| {
        let hlc = Arc::new(HybridLogicalClock::new_with_fixed_time(1, 1000));

        let hlc1 = Arc::clone(&hlc);
        let hlc2 = Arc::clone(&hlc);

        let t1 = thread::spawn(move || hlc1.now().unwrap());
        let t2 = thread::spawn(move || hlc2.now().unwrap());

        let ts1 = t1.join().unwrap();
        let ts2 = t2.join().unwrap();

        // Timestamps must be unique
        assert_ne!(
            ts1, ts2,
            "Concurrent now() calls must produce unique timestamps: {ts1:?} vs {ts2:?}"
        );
    });
}

#[test]
fn loom_hlc_concurrent_now_monotonic() {
    loom::model(|| {
        let hlc = Arc::new(HybridLogicalClock::new_with_fixed_time(1, 1000));

        let hlc1 = Arc::clone(&hlc);
        let hlc2 = Arc::clone(&hlc);

        let t1 = thread::spawn(move || hlc1.now().unwrap());
        let t2 = thread::spawn(move || hlc2.now().unwrap());

        let ts1 = t1.join().unwrap();
        let ts2 = t2.join().unwrap();

        // After all threads complete, a sequential call must exceed both
        let ts3 = hlc.now().unwrap();
        assert!(
            ts3 > ts1,
            "Sequential now() must exceed thread 1: {ts3:?} vs {ts1:?}"
        );
        assert!(
            ts3 > ts2,
            "Sequential now() must exceed thread 2: {ts3:?} vs {ts2:?}"
        );
    });
}

#[test]
fn loom_hlc_concurrent_receive_ordering() {
    loom::model(|| {
        let hlc = Arc::new(HybridLogicalClock::new_with_fixed_time(1, 1000));

        let remote = tensor_chain::HLCTimestamp::new(2000, 5, 99);

        let hlc1 = Arc::clone(&hlc);
        let hlc2 = Arc::clone(&hlc);
        let remote_copy = remote;

        let t1 = thread::spawn(move || hlc1.now().unwrap());
        let t2 = thread::spawn(move || hlc2.receive(&remote_copy).unwrap());

        let ts1 = t1.join().unwrap();
        let ts2 = t2.join().unwrap();

        // receive() result must be strictly after the remote timestamp
        assert!(
            ts2 > remote,
            "receive() result must exceed remote: {ts2:?} vs {remote:?}"
        );

        // Both timestamps must be unique
        assert_ne!(
            ts1, ts2,
            "now() and receive() must produce unique timestamps"
        );
    });
}

#[test]
fn loom_hlc_concurrent_drift_injection() {
    loom::model(|| {
        let hlc = Arc::new(HybridLogicalClock::new_with_fixed_time(1, 1000));

        let hlc1 = Arc::clone(&hlc);
        let hlc2 = Arc::clone(&hlc);

        let t1 = thread::spawn(move || {
            hlc1.set_drift_offset(500);
        });

        let t2 = thread::spawn(move || hlc2.now().unwrap());

        t1.join().unwrap();
        let ts = t2.join().unwrap();

        // Timestamp must be valid regardless of drift timing
        assert!(
            ts.wall_ms() >= 1000,
            "wall_ms must be at least base time: got {}",
            ts.wall_ms()
        );
    });
}
