#![no_main]
use libfuzzer_sys::fuzz_target;
use query_router::QueryRouter;
use tensor_store::TensorStore;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        if s.len() > 4096 {
            return;
        }
        let store = TensorStore::new();
        let router = QueryRouter::with_shared_store(store);
        // Execute through the parsed path (synchronous wrapper over async)
        let _ = router.execute_parsed(s);
    }
});
