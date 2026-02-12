#![no_main]
use libfuzzer_sys::fuzz_target;
use neumann_shell::{Shell, ShellConfig};

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        if s.len() > 4096 {
            return;
        }
        let config = ShellConfig {
            history_file: None,
            no_color: true,
            no_boot: true,
            quiet: true,
            ..ShellConfig::default()
        };
        let mut shell = Shell::with_config(config);
        // Must not panic regardless of input
        let _ = shell.execute(s);
    }
});
