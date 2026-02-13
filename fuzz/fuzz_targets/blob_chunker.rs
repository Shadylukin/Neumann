#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_blob::{compute_hash, Chunk, Chunker};

#[derive(Arbitrary, Debug)]
struct ChunkerInput {
    chunk_size_raw: u16,
    data: Vec<u8>,
}

fuzz_target!(|input: ChunkerInput| {
    if input.data.len() > 1_048_576 {
        return;
    }

    // Chunk size between 1 and 65535
    let chunk_size = (input.chunk_size_raw as usize).max(1);
    let chunker = Chunker::new(chunk_size);

    // Verify chunk count prediction matches reality
    let predicted = chunker.chunk_count(input.data.len());
    let chunks: Vec<Chunk> = chunker.chunk(&input.data).collect();
    assert_eq!(chunks.len(), predicted);

    // Verify reassembly produces original data
    let reassembled: Vec<u8> = chunks.iter().flat_map(|c| c.data.iter().copied()).collect();
    assert_eq!(reassembled, input.data);

    // Verify each chunk hash matches compute_hash
    for chunk in &chunks {
        let expected_hash = compute_hash(&chunk.data);
        assert_eq!(chunk.hash, expected_hash);
        assert_eq!(chunk.size, chunk.data.len());
    }

    // Verify overall hash is deterministic
    let hash1 = compute_hash(&input.data);
    let hash2 = compute_hash(&input.data);
    assert_eq!(hash1, hash2);
});
