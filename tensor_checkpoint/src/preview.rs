use crate::state::{DestructiveOp, OperationPreview};

pub struct PreviewGenerator {
    sample_size: usize,
}

impl PreviewGenerator {
    pub fn new(sample_size: usize) -> Self {
        Self { sample_size }
    }

    pub fn generate(&self, op: &DestructiveOp, sample_data: Vec<String>) -> OperationPreview {
        let summary = self.format_summary(op);
        let affected_count = op.affected_count();

        let truncated_sample: Vec<String> =
            sample_data.into_iter().take(self.sample_size).collect();

        OperationPreview::new(summary, truncated_sample, affected_count)
    }

    fn format_summary(&self, op: &DestructiveOp) -> String {
        match op {
            DestructiveOp::Delete { table, row_count } => {
                format!("Will delete {row_count} row(s) from table '{table}'")
            },
            DestructiveOp::DropTable { table, row_count } => {
                format!("Will drop table '{table}' containing {row_count} row(s)")
            },
            DestructiveOp::DropIndex { table, column } => {
                format!("Will drop index on column '{column}' in table '{table}'")
            },
            DestructiveOp::NodeDelete {
                node_id,
                edge_count,
            } => {
                format!("Will delete node {node_id} and {edge_count} connected edge(s)")
            },
            DestructiveOp::EdgeDelete { edge_id } => {
                format!("Will delete edge {edge_id}")
            },
            DestructiveOp::EmbedDelete { key } => {
                format!("Will delete embedding with key '{key}'")
            },
            DestructiveOp::VaultDelete { key } => {
                format!("Will delete vault secret with key '{key}'")
            },
            DestructiveOp::BlobDelete { artifact_id, size } => {
                let size_str = format_bytes(*size);
                format!("Will delete blob artifact '{artifact_id}' ({size_str})")
            },
            DestructiveOp::CacheClear { entry_count } => {
                format!("Will clear cache with {entry_count} entries")
            },
        }
    }
}

impl Default for PreviewGenerator {
    fn default() -> Self {
        Self::new(5)
    }
}

fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

pub fn format_warning(op: &DestructiveOp) -> String {
    match op {
        DestructiveOp::Delete { table, row_count } => {
            format!("WARNING: About to delete {row_count} row(s) from table '{table}'")
        },
        DestructiveOp::DropTable { table, row_count } => {
            format!("WARNING: About to drop table '{table}' with {row_count} row(s)")
        },
        DestructiveOp::DropIndex { table, column } => {
            format!("WARNING: About to drop index on '{column}' in table '{table}'")
        },
        DestructiveOp::NodeDelete {
            node_id,
            edge_count,
        } => {
            format!("WARNING: About to delete node {node_id} and {edge_count} connected edge(s)")
        },
        DestructiveOp::EdgeDelete { edge_id } => {
            format!("WARNING: About to delete edge {edge_id}")
        },
        DestructiveOp::EmbedDelete { key } => {
            format!("WARNING: About to delete embedding '{key}'")
        },
        DestructiveOp::VaultDelete { key } => {
            format!("WARNING: About to delete vault secret '{key}'")
        },
        DestructiveOp::BlobDelete { artifact_id, size } => {
            let size_str = format_bytes(*size);
            format!("WARNING: About to delete blob '{artifact_id}' ({size_str})")
        },
        DestructiveOp::CacheClear { entry_count } => {
            format!("WARNING: About to clear cache with {entry_count} entries")
        },
    }
}

pub fn format_confirmation_prompt(op: &DestructiveOp, preview: &OperationPreview) -> String {
    let mut output = String::new();

    output.push_str(&format_warning(op));
    output.push('\n');
    output.push_str(&preview.summary);
    output.push('\n');

    if !preview.sample_data.is_empty() {
        output.push_str("\nAffected data sample:\n");
        for (i, item) in preview.sample_data.iter().enumerate() {
            output.push_str(&format!("  {}. {}\n", i + 1, item));
        }
        if preview.affected_count > preview.sample_data.len() {
            output.push_str(&format!(
                "  ... and {} more\n",
                preview.affected_count - preview.sample_data.len()
            ));
        }
    }

    output.push_str("\nType 'yes' to proceed, anything else to cancel: ");

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 bytes");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_delete_preview() {
        let gen = PreviewGenerator::new(3);
        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 10,
        };

        let sample = vec![
            "id=1, name='Alice'".to_string(),
            "id=2, name='Bob'".to_string(),
        ];

        let preview = gen.generate(&op, sample);
        assert_eq!(preview.summary, "Will delete 10 row(s) from table 'users'");
        assert_eq!(preview.sample_data.len(), 2);
        assert_eq!(preview.affected_count, 10);
    }

    #[test]
    fn test_drop_table_preview() {
        let gen = PreviewGenerator::default();
        let op = DestructiveOp::DropTable {
            table: "temp_data".to_string(),
            row_count: 1000,
        };

        let preview = gen.generate(&op, vec![]);
        assert_eq!(
            preview.summary,
            "Will drop table 'temp_data' containing 1000 row(s)"
        );
    }

    #[test]
    fn test_sample_truncation() {
        let gen = PreviewGenerator::new(2);
        let op = DestructiveOp::Delete {
            table: "big_table".to_string(),
            row_count: 100,
        };

        let sample = vec![
            "row1".to_string(),
            "row2".to_string(),
            "row3".to_string(),
            "row4".to_string(),
        ];

        let preview = gen.generate(&op, sample);
        assert_eq!(preview.sample_data.len(), 2);
        assert_eq!(preview.affected_count, 100);
    }

    #[test]
    fn test_confirmation_prompt() {
        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 5,
        };
        let preview = OperationPreview::new(
            "Will delete 5 rows".to_string(),
            vec!["id=1".to_string(), "id=2".to_string()],
            5,
        );

        let prompt = format_confirmation_prompt(&op, &preview);
        assert!(prompt.contains("WARNING"));
        assert!(prompt.contains("Affected data sample"));
        assert!(prompt.contains("... and 3 more"));
        assert!(prompt.contains("Type 'yes'"));
    }

    #[test]
    fn test_confirmation_prompt_no_more() {
        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 2,
        };
        let preview = OperationPreview::new(
            "Will delete 2 rows".to_string(),
            vec!["id=1".to_string(), "id=2".to_string()],
            2,
        );

        let prompt = format_confirmation_prompt(&op, &preview);
        assert!(!prompt.contains("... and"));
    }

    #[test]
    fn test_confirmation_prompt_empty_sample() {
        let op = DestructiveOp::Delete {
            table: "users".to_string(),
            row_count: 0,
        };
        let preview = OperationPreview::new("Will delete 0 rows".to_string(), vec![], 0);

        let prompt = format_confirmation_prompt(&op, &preview);
        assert!(!prompt.contains("Affected data sample"));
    }

    #[test]
    fn test_format_warning_all_ops() {
        let ops = vec![
            DestructiveOp::Delete {
                table: "t".to_string(),
                row_count: 1,
            },
            DestructiveOp::DropTable {
                table: "t".to_string(),
                row_count: 1,
            },
            DestructiveOp::DropIndex {
                table: "t".to_string(),
                column: "c".to_string(),
            },
            DestructiveOp::NodeDelete {
                node_id: 1,
                edge_count: 2,
            },
            DestructiveOp::EdgeDelete { edge_id: 1 },
            DestructiveOp::EmbedDelete {
                key: "k".to_string(),
            },
            DestructiveOp::VaultDelete {
                key: "k".to_string(),
            },
            DestructiveOp::BlobDelete {
                artifact_id: "a".to_string(),
                size: 1024,
            },
            DestructiveOp::CacheClear { entry_count: 10 },
        ];

        for op in ops {
            let warning = format_warning(&op);
            assert!(warning.starts_with("WARNING:"));
        }
    }

    #[test]
    fn test_all_op_previews() {
        let gen = PreviewGenerator::new(5);

        let ops: Vec<DestructiveOp> = vec![
            DestructiveOp::DropIndex {
                table: "users".to_string(),
                column: "email".to_string(),
            },
            DestructiveOp::NodeDelete {
                node_id: 42,
                edge_count: 5,
            },
            DestructiveOp::EdgeDelete { edge_id: 42 },
            DestructiveOp::EmbedDelete {
                key: "doc1".to_string(),
            },
            DestructiveOp::VaultDelete {
                key: "api_key".to_string(),
            },
            DestructiveOp::BlobDelete {
                artifact_id: "blob-123".to_string(),
                size: 1073741824, // 1 GB
            },
            DestructiveOp::CacheClear { entry_count: 100 },
        ];

        for op in ops {
            let preview = gen.generate(&op, vec![]);
            assert!(!preview.summary.is_empty());
        }
    }

    #[test]
    fn test_format_bytes_edge_cases() {
        assert_eq!(format_bytes(0), "0 bytes");
        assert_eq!(format_bytes(1), "1 bytes");
        assert_eq!(format_bytes(1023), "1023 bytes");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.00 MB");
        assert_eq!(format_bytes(5 * 1024 * 1024 * 1024), "5.00 GB");
    }
}
