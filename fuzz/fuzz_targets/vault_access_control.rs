#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use query_router::QueryRouter;
use tensor_store::TensorStore;

#[derive(Arbitrary, Debug)]
enum VaultOp {
    Put { key: String, value: String },
    Get { key: String },
    Delete { key: String },
    Grant { entity: String, key: String },
    Revoke { entity: String, key: String },
    SetIdentity { identity: String },
    List,
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    ops: Vec<VaultOp>,
}

fuzz_target!(|input: FuzzInput| {
    if input.ops.len() > 50 {
        return;
    }

    let store = TensorStore::new();
    let mut router = QueryRouter::with_shared_store(store);
    let _ = router.init_vault(b"fuzz-master-key-32-bytes-long!!!");

    for op in &input.ops {
        match op {
            VaultOp::SetIdentity { identity } => {
                if identity.len() <= 128 {
                    let _ = router.execute(&format!("SET IDENTITY {identity}"));
                }
            },
            VaultOp::Put { key, value } => {
                if key.len() <= 128 && value.len() <= 512 && !key.is_empty() {
                    let _ = router.execute(&format!("VAULT PUT '{key}' '{value}'"));
                }
            },
            VaultOp::Get { key } => {
                if key.len() <= 128 && !key.is_empty() {
                    let _ = router.execute(&format!("VAULT GET '{key}'"));
                }
            },
            VaultOp::Delete { key } => {
                if key.len() <= 128 && !key.is_empty() {
                    let _ = router.execute(&format!("VAULT DELETE '{key}'"));
                }
            },
            VaultOp::Grant { entity, key } => {
                if entity.len() <= 128 && key.len() <= 128 && !entity.is_empty() && !key.is_empty()
                {
                    let _ = router.execute(&format!("VAULT GRANT '{entity}' '{key}'"));
                }
            },
            VaultOp::Revoke { entity, key } => {
                if entity.len() <= 128 && key.len() <= 128 && !entity.is_empty() && !key.is_empty()
                {
                    let _ = router.execute(&format!("VAULT REVOKE '{entity}' '{key}'"));
                }
            },
            VaultOp::List => {
                let _ = router.execute("VAULT LIST");
            },
        }
    }
});
