#![no_main]
use libfuzzer_sys::fuzz_target;
use neumann_shell::{Shell, ShellConfig};
fuzz_target!(|data: &[u8]| {
    if data.len() > 8192 {
        return;
    }

    // Convert fuzz data to lines that look like shell commands
    if let Ok(s) = std::str::from_utf8(data) {
        let config = ShellConfig {
            history_file: None,
            no_color: true,
            no_boot: true,
            quiet: true,
            ..ShellConfig::default()
        };
        let mut shell = Shell::with_config(config);

        // Execute each line as a command (simulates WAL replay)
        for line in s.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            // Must not panic regardless of input
            let _ = shell.execute(trimmed);
        }
    }
});
