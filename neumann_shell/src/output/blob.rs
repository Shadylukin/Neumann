// SPDX-License-Identifier: MIT OR Apache-2.0
//! Blob and binary data formatting.

#![allow(clippy::needless_borrows_for_generic_args)]

use crate::style::{styled, Theme};
use std::fmt::Write;

// Display constants
const BLOB_TEXT_DISPLAY_LIMIT: usize = 4096;
const BLOB_HEX_PREVIEW_BYTES: usize = 64;
const BLOB_HEX_BYTES_PER_LINE: usize = 16;

/// Formats blob data for display.
#[must_use]
pub fn format_blob(data: &[u8], theme: &Theme) -> String {
    let size = data.len();

    // Try UTF-8 text display (up to 4KB)
    if size <= BLOB_TEXT_DISPLAY_LIMIT {
        if let Ok(s) = std::str::from_utf8(data) {
            if s.chars()
                .all(|c| !c.is_control() || c == '\n' || c == '\t' || c == '\r')
            {
                return styled(s, theme.string);
            }
        }
    }

    // Large text files: show preview
    if size > BLOB_TEXT_DISPLAY_LIMIT {
        if let Ok(s) = std::str::from_utf8(data) {
            if s.chars()
                .all(|c| !c.is_control() || c == '\n' || c == '\t' || c == '\r')
            {
                let preview_end = s
                    .char_indices()
                    .take_while(|(i, _)| *i < BLOB_TEXT_DISPLAY_LIMIT)
                    .last()
                    .map_or(BLOB_TEXT_DISPLAY_LIMIT, |(i, c)| i + c.len_utf8());

                return format!(
                    "{}\n{} ({} more bytes, {} total)",
                    styled(&s[..preview_end], theme.string),
                    styled("...", theme.muted),
                    styled(size - preview_end, theme.number),
                    styled(size, theme.number)
                );
            }
        }
    }

    // Binary data: hex dump preview
    let mut output = format!(
        "{}\n\n",
        styled(format!("<binary data: {size} bytes>"), theme.muted)
    );
    output.push_str(&format_hex_dump(data, BLOB_HEX_PREVIEW_BYTES, theme));

    if size > BLOB_HEX_PREVIEW_BYTES {
        let _ = writeln!(
            output,
            "{} ({} more bytes)",
            styled("...", theme.muted),
            styled(size - BLOB_HEX_PREVIEW_BYTES, theme.number)
        );
    }
    output
}

/// Formats a hex dump of binary data.
#[must_use]
pub fn format_hex_dump(data: &[u8], max_bytes: usize, theme: &Theme) -> String {
    let bytes_to_show = data.len().min(max_bytes);
    let mut output = String::new();

    for (i, chunk) in data[..bytes_to_show]
        .chunks(BLOB_HEX_BYTES_PER_LINE)
        .enumerate()
    {
        // Offset (styled)
        let _ = write!(
            output,
            "{}  ",
            styled(format!("{:08x}", i * BLOB_HEX_BYTES_PER_LINE), theme.muted)
        );

        // Hex bytes with midpoint spacing
        for (j, byte) in chunk.iter().enumerate() {
            let _ = write!(output, "{} ", styled(format!("{byte:02x}"), theme.number));
            if j == 7 {
                output.push(' ');
            }
        }

        // Padding for incomplete lines
        for j in chunk.len()..BLOB_HEX_BYTES_PER_LINE {
            output.push_str("   ");
            if j == 7 {
                output.push(' ');
            }
        }

        // ASCII representation
        output.push_str(" |");
        for byte in chunk {
            let c = if byte.is_ascii_graphic() || *byte == b' ' {
                *byte as char
            } else {
                '.'
            };
            output.push(c);
        }
        output.push_str("|\n");
    }
    output
}

/// Formats artifact info for display.
#[must_use]
pub fn format_artifact_info(info: &query_router::ArtifactInfoResult, theme: &Theme) -> String {
    let mut lines = vec![
        format!(
            "{}: {}",
            styled("Artifact", theme.header),
            styled(&info.id, theme.id)
        ),
        format!("  Filename: {}", styled(&info.filename, theme.string)),
        format!("  Type: {}", styled(&info.content_type, theme.label)),
        format!("  Size: {} bytes", styled(info.size, theme.number)),
        format!("  Checksum: {}", styled(&info.checksum, theme.muted)),
        format!("  Chunks: {}", styled(info.chunk_count, theme.number)),
        format!("  Created: {}", styled(&info.created, theme.muted)),
        format!("  Modified: {}", styled(&info.modified, theme.muted)),
        format!("  Creator: {}", styled(&info.created_by, theme.label)),
    ];

    if !info.tags.is_empty() {
        let tags: Vec<String> = info.tags.iter().map(|t| styled(t, theme.keyword)).collect();
        lines.push(format!("  Tags: {}", tags.join(", ")));
    }

    if !info.linked_to.is_empty() {
        let linked: Vec<String> = info.linked_to.iter().map(|l| styled(l, theme.id)).collect();
        lines.push(format!("  Links: {}", linked.join(", ")));
    }

    if !info.custom.is_empty() {
        lines.push(format!("  {}:", styled("Metadata", theme.header)));
        for (k, v) in &info.custom {
            lines.push(format!(
                "    {}: {}",
                styled(k, theme.keyword),
                styled(v, theme.string)
            ));
        }
    }

    lines.join("\n")
}

/// Formats artifact list for display.
#[must_use]
pub fn format_artifact_list(ids: &[String], theme: &Theme) -> String {
    if ids.is_empty() {
        styled("(no artifacts)", theme.muted)
    } else {
        ids.iter()
            .map(|id| styled(id, theme.id))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Formats blob statistics for display.
#[must_use]
pub fn format_blob_stats(stats: &query_router::BlobStatsResult, theme: &Theme) -> String {
    format!(
        "{}\n\
         Artifacts: {}\n\
         Chunks: {}\n\
         Total bytes: {}\n\
         Unique bytes: {}\n\
         Dedup ratio: {}\n\
         Orphaned chunks: {}",
        styled("Blob Storage Statistics:", theme.header),
        styled(stats.artifact_count, theme.number),
        styled(stats.chunk_count, theme.number),
        styled(stats.total_bytes, theme.number),
        styled(stats.unique_bytes, theme.number),
        styled(format!("{:.1}%", stats.dedup_ratio * 100.0), theme.number),
        styled(stats.orphaned_chunks, theme.number)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_blob_text() {
        let theme = Theme::plain();
        let result = format_blob(b"Hello, World!", &theme);
        assert!(result.contains("Hello, World!"));
    }

    #[test]
    fn test_format_blob_binary() {
        let theme = Theme::plain();
        let result = format_blob(&[0x00, 0x01, 0xFF, 0xFE], &theme);
        assert!(result.contains("binary data"));
        assert!(result.contains("00"));
        assert!(result.contains("ff"));
    }

    #[test]
    fn test_format_hex_dump() {
        let theme = Theme::plain();
        let result = format_hex_dump(&[0xDE, 0xAD, 0xBE, 0xEF], 16, &theme);
        assert!(result.contains("de"));
        assert!(result.contains("ad"));
        assert!(result.contains("be"));
        assert!(result.contains("ef"));
    }

    #[test]
    fn test_format_artifact_list_empty() {
        let theme = Theme::plain();
        let result = format_artifact_list(&[], &theme);
        assert!(result.contains("no artifacts"));
    }

    #[test]
    fn test_format_artifact_list_with_items() {
        let theme = Theme::plain();
        let result = format_artifact_list(&["id1".to_string(), "id2".to_string()], &theme);
        assert!(result.contains("id1"));
        assert!(result.contains("id2"));
    }

    #[test]
    fn test_format_blob_stats() {
        let theme = Theme::plain();
        let stats = query_router::BlobStatsResult {
            artifact_count: 10,
            chunk_count: 100,
            total_bytes: 10000,
            unique_bytes: 5000,
            dedup_ratio: 0.5,
            orphaned_chunks: 2,
        };
        let result = format_blob_stats(&stats, &theme);
        assert!(result.contains("10"));
        assert!(result.contains("100"));
        assert!(result.contains("50.0%"));
    }
}
