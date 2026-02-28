#![no_main]
use libfuzzer_sys::fuzz_target;
use query_router::QueryRouter;
use tensor_store::TensorStore;

fuzz_target!(|data: &[u8]| {
    if data.len() > 65536 {
        return;
    }

    // Test checkpoint operations through the query router
    let store = TensorStore::new();
    let router = QueryRouter::with_shared_store(store);

    // Insert some data first
    let _ = router.execute("CREATE TABLE test (name TEXT, val INT)");
    let _ = router.execute("INSERT INTO test (name, val) VALUES ('a', 1)");
    let _ = router.execute("NODE CREATE label {k: 'value'}");

    // Try checkpoint and rollback
    let _ = router.execute("CHECKPOINT CREATE");
    let _ = router.execute("CHECKPOINT LIST");

    // Also try to load corrupted snapshot data as query input
    if let Ok(s) = std::str::from_utf8(data) {
        if s.len() <= 4096 {
            // Try arbitrary CHECKPOINT/ROLLBACK commands
            let _ = router.execute(s);
        }
    }
});
