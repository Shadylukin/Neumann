#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_blob::{BlobConfig, BlobStore, PutOptions};
use tensor_store::TensorStore;

#[derive(Arbitrary, Debug)]
struct MetadataInput {
    name: String,
    content_type: Option<String>,
    created_by: Option<String>,
    tags: Vec<String>,
    meta_keys: Vec<String>,
    meta_values: Vec<String>,
}

fuzz_target!(|input: MetadataInput| {
    if input.name.len() > 256
        || input.tags.len() > 32
        || input.meta_keys.len() > 32
        || input.name.is_empty()
    {
        return;
    }

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let store = TensorStore::new();
        let config = BlobConfig::default();
        let blob_store = match BlobStore::new(store, config).await {
            Ok(bs) => bs,
            Err(_) => return,
        };

        let mut opts = PutOptions::new();
        if let Some(ct) = &input.content_type {
            if ct.len() <= 128 {
                opts = opts.with_content_type(ct.as_str());
            }
        }
        if let Some(cb) = &input.created_by {
            if cb.len() <= 128 {
                opts = opts.with_created_by(cb.as_str());
            }
        }
        for tag in &input.tags {
            if tag.len() <= 64 {
                opts = opts.with_tag(tag.as_str());
            }
        }
        let pairs = input.meta_keys.len().min(input.meta_values.len());
        for i in 0..pairs {
            if input.meta_keys[i].len() <= 64 && input.meta_values[i].len() <= 256 {
                opts = opts.with_meta(&input.meta_keys[i], &input.meta_values[i]);
            }
        }

        let data = b"fuzz test content";
        // Put should not panic
        let _ = blob_store.put(&input.name, data, opts).await;
    });
});
