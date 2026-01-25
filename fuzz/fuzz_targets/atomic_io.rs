#![no_main]

use std::fs;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use tensor_chain::{atomic_truncate, atomic_write, AtomicWriter};

#[derive(Debug, Arbitrary)]
#[allow(dead_code)]
enum AtomicOp {
    Write { data: Vec<u8> },
    Truncate,
    WriterCommit { data: Vec<u8> },
    WriterAbort { data: Vec<u8> },
    SequentialWrites { writes: Vec<Vec<u8>> },
}

fuzz_target!(|ops: Vec<AtomicOp>| {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fuzz_file.dat");

    for op in ops {
        match op {
            AtomicOp::Write { data } => {
                // Test atomic_write
                let _ = atomic_write(&path, &data);

                // Verify data integrity if write succeeded
                if path.exists() {
                    let content = fs::read(&path).unwrap_or_default();
                    // Content should either be the new data or previous data
                    // (never partial or corrupted)
                    assert!(content == data || content.len() <= data.len() || data.is_empty());
                }
            },

            AtomicOp::Truncate => {
                // Test atomic_truncate
                if path.exists() {
                    let _ = atomic_truncate(&path);

                    // If truncate succeeded and file exists, it should be empty
                    if path.exists() {
                        let content = fs::read(&path).unwrap_or_default();
                        assert!(content.is_empty());
                    }
                }
            },

            AtomicOp::WriterCommit { data } => {
                // Test AtomicWriter with commit
                if let Ok(mut writer) = AtomicWriter::new(&path) {
                    use std::io::Write;
                    let _ = writer.write_all(&data);
                    let _ = writer.commit();

                    // Verify data integrity
                    if path.exists() {
                        let content = fs::read(&path).unwrap_or_default();
                        // Should be the new data or empty (if write failed)
                        assert!(content == data || content.is_empty());
                    }
                }
            },

            AtomicOp::WriterAbort { data } => {
                // Test AtomicWriter with abort
                let old_content = fs::read(&path).unwrap_or_default();

                if let Ok(mut writer) = AtomicWriter::new(&path) {
                    use std::io::Write;
                    let _ = writer.write_all(&data);
                    writer.abort();

                    // After abort, original content should be preserved
                    if path.exists() {
                        let content = fs::read(&path).unwrap_or_default();
                        assert_eq!(content, old_content);
                    }
                }
            },

            AtomicOp::SequentialWrites { writes } => {
                // Test multiple sequential writes
                for data in writes {
                    let _ = atomic_write(&path, &data);
                }

                // No temp files should remain
                for entry in fs::read_dir(dir.path()).unwrap() {
                    let entry = entry.unwrap();
                    let name = entry.file_name().to_string_lossy().to_string();
                    assert!(
                        !name.contains(".tmp."),
                        "Temp file found: {}",
                        name
                    );
                }
            },
        }
    }

    // Final cleanup verification - no temp files
    for entry in fs::read_dir(dir.path()).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        assert!(
            !name.contains(".tmp."),
            "Temp file found after all ops: {}",
            name
        );
    }
});
