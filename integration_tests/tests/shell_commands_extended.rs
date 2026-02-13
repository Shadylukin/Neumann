// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Extended shell command integration tests.
//!
//! Tests shell built-in commands, SAVE/LOAD snapshots, WAL operations,
//! doctor diagnostics, and error recovery through the parser-based execute path.

use neumann_shell::{CommandResult, Shell};
use tempfile::TempDir;

fn create_test_shell() -> Shell {
    Shell::new()
}

#[test]
fn test_shell_help_returns_help_variant() {
    let mut shell = create_test_shell();
    let result = shell.execute("help");
    match result {
        CommandResult::Help(text) => {
            assert!(!text.is_empty(), "help text should not be empty");
            let lower = text.to_lowercase();
            assert!(
                lower.contains("select") || lower.contains("insert") || lower.contains("create"),
                "help text should reference basic commands"
            );
        },
        other => panic!("expected Help variant, got {:?}", other),
    }
}

#[test]
fn test_shell_exit_returns_exit_variant() {
    let mut shell = create_test_shell();

    assert_eq!(shell.execute("exit"), CommandResult::Exit);
    assert_eq!(shell.execute("quit"), CommandResult::Exit);
    assert_eq!(shell.execute("\\q"), CommandResult::Exit);
}

#[test]
fn test_shell_tables_after_create() {
    let mut shell = create_test_shell();

    let create_result = shell.execute("CREATE TABLE shell_test_tbl (id INT, name TEXT)");
    assert!(
        !matches!(create_result, CommandResult::Error(_)),
        "CREATE TABLE should succeed: {:?}",
        create_result
    );

    let tables_result = shell.execute("tables");
    match tables_result {
        CommandResult::Output(text) => {
            assert!(
                text.contains("shell_test_tbl"),
                "tables output should include created table, got: {}",
                text
            );
        },
        other => panic!("expected Output from tables, got {:?}", other),
    }
}

#[test]
fn test_shell_empty_input() {
    let mut shell = create_test_shell();
    assert_eq!(shell.execute(""), CommandResult::Empty);
    assert_eq!(shell.execute("   "), CommandResult::Empty);
    assert_eq!(shell.execute("\t\n"), CommandResult::Empty);
}

#[test]
fn test_shell_error_recovery() {
    let mut shell = create_test_shell();

    let bad = shell.execute("TOTALLY_INVALID_COMMAND foo bar");
    assert!(
        matches!(bad, CommandResult::Error(_)),
        "invalid command should return Error"
    );

    // Shell should still work after error -- NODE CREATE uses parser syntax
    let node_result = shell.execute("NODE CREATE test_item");
    assert!(
        matches!(node_result, CommandResult::Output(_)),
        "NODE CREATE should work after error: {:?}",
        node_result
    );

    let bad2 = shell.execute("XYZZY_NONSENSE");
    assert!(matches!(bad2, CommandResult::Error(_)));

    // Parser EMBED syntax: EMBED STORE 'key' [values]
    let embed_result = shell.execute("EMBED STORE 'recovery_vec' [0.1, 0.2, 0.3]");
    assert!(
        !matches!(embed_result, CommandResult::Error(_)),
        "EMBED STORE should work after second error: {:?}",
        embed_result
    );
}

#[test]
fn test_shell_save_and_load() {
    let tmp = TempDir::new().unwrap();
    let snap_path = tmp.path().join("test_snapshot.bin");
    let snap_str = snap_path.to_str().unwrap();

    let mut shell1 = create_test_shell();
    shell1.execute("CREATE TABLE snap_tbl (id INT, val TEXT)");
    let insert1 = shell1.execute("INSERT INTO snap_tbl (id, val) VALUES (1, 'hello')");
    assert!(
        !matches!(insert1, CommandResult::Error(_)),
        "INSERT should succeed: {:?}",
        insert1
    );
    shell1.execute("INSERT INTO snap_tbl (id, val) VALUES (2, 'world')");

    let save_result = shell1.execute(&format!("SAVE {snap_str}"));
    match &save_result {
        CommandResult::Output(text) => {
            assert!(
                text.to_lowercase().contains("saved") || text.contains("snapshot"),
                "SAVE output should confirm success: {}",
                text
            );
        },
        CommandResult::Error(e) => panic!("SAVE failed: {}", e),
        other => panic!("unexpected SAVE result: {:?}", other),
    }

    let mut shell2 = Shell::new();
    let load_result = shell2.execute(&format!("LOAD {snap_str}"));
    match &load_result {
        CommandResult::Output(text) => {
            assert!(
                text.to_lowercase().contains("loaded") || text.contains("snapshot"),
                "LOAD output should confirm success: {}",
                text
            );
        },
        CommandResult::Error(e) => panic!("LOAD failed: {}", e),
        other => panic!("unexpected LOAD result: {:?}", other),
    }

    let select = shell2.execute("SELECT * FROM snap_tbl");
    match select {
        CommandResult::Output(text) => {
            assert!(text.contains("hello"), "loaded data should contain 'hello'");
            assert!(text.contains("world"), "loaded data should contain 'world'");
        },
        other => panic!("expected Output from SELECT after LOAD, got {:?}", other),
    }
}

#[test]
fn test_shell_save_compressed_and_load() {
    let tmp = TempDir::new().unwrap();
    let snap_path = tmp.path().join("compressed_snapshot.bin");
    let snap_str = snap_path.to_str().unwrap();

    let mut shell1 = create_test_shell();
    shell1.execute("CREATE TABLE comp_tbl (name TEXT)");
    shell1.execute("INSERT INTO comp_tbl (name) VALUES ('compressed_test')");
    // Parser EMBED syntax required for execute_parsed
    shell1.execute("EMBED STORE 'comp_key' [1.0, 0.5, 0.3, 0.1]");

    let save_result = shell1.execute(&format!("SAVE COMPRESSED {snap_str}"));
    match &save_result {
        CommandResult::Output(text) => {
            assert!(
                text.to_lowercase().contains("saved") || text.contains("compressed"),
                "SAVE COMPRESSED should confirm: {}",
                text
            );
        },
        CommandResult::Error(e) => panic!("SAVE COMPRESSED failed: {}", e),
        other => panic!("unexpected SAVE COMPRESSED result: {:?}", other),
    }

    let mut shell2 = Shell::new();
    let load_result = shell2.execute(&format!("LOAD {snap_str}"));
    match &load_result {
        CommandResult::Output(text) => {
            assert!(
                text.to_lowercase().contains("loaded"),
                "LOAD should confirm: {}",
                text
            );
        },
        CommandResult::Error(e) => panic!("LOAD compressed snapshot failed: {}", e),
        other => panic!("unexpected LOAD result: {:?}", other),
    }

    // Verify embedding survived the compressed save/load cycle
    let embed_get = shell2.execute("EMBED GET 'comp_key'");
    match embed_get {
        CommandResult::Output(text) => {
            assert!(
                !text.is_empty(),
                "EMBED GET should return embedding data after compressed load"
            );
        },
        other => panic!(
            "expected Output from EMBED GET after compressed LOAD, got {:?}",
            other
        ),
    }
}

#[test]
fn test_shell_doctor_command() {
    let mut shell = create_test_shell();

    shell.execute("CREATE TABLE doctor_tbl (x INT)");
    shell.execute("INSERT INTO doctor_tbl (x) VALUES (1)");
    shell.execute("NODE CREATE person name='DocTest'");

    let result = shell.execute("doctor");
    match result {
        CommandResult::Output(text) => {
            assert!(!text.is_empty(), "doctor output should not be empty");
        },
        other => panic!("expected Output from doctor, got {:?}", other),
    }
}

#[test]
fn test_shell_clear_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("clear");
    match result {
        CommandResult::Output(text) => {
            assert!(
                text.contains("\x1B[2J") || text.contains("\x1b[2J"),
                "clear should produce ANSI clear sequence"
            );
        },
        other => panic!("expected Output from clear, got {:?}", other),
    }
}

#[test]
fn test_shell_query_through_execute() {
    let mut shell = create_test_shell();

    shell.execute("CREATE TABLE qtest (id INT, color TEXT)");
    shell.execute("INSERT INTO qtest (id, color) VALUES (1, 'red')");
    shell.execute("INSERT INTO qtest (id, color) VALUES (2, 'blue')");

    let result = shell.execute("SELECT * FROM qtest");
    match result {
        CommandResult::Output(text) => {
            assert!(text.contains("red"), "should contain 'red'");
            assert!(text.contains("blue"), "should contain 'blue'");
        },
        CommandResult::Error(e) => panic!("SELECT through shell failed: {}", e),
        other => panic!("unexpected SELECT result: {:?}", other),
    }

    // Graph node through shell (parser syntax)
    let node_result = shell.execute("NODE CREATE item { name: 'Widget' }");
    assert!(
        matches!(node_result, CommandResult::Output(_)),
        "NODE CREATE should succeed: {:?}",
        node_result
    );

    // Embedding through shell (parser EMBED STORE syntax)
    let embed_result = shell.execute("EMBED STORE 'widget_vec' [0.1, 0.2, 0.3, 0.4]");
    assert!(
        !matches!(embed_result, CommandResult::Error(_)),
        "EMBED STORE should succeed: {:?}",
        embed_result
    );
}

#[test]
fn test_shell_wal_status_without_load() {
    let mut shell = create_test_shell();

    let result = shell.execute("wal status");
    match result {
        CommandResult::Output(text) => {
            assert!(
                text.contains("not active") || text.contains("WAL"),
                "wal status should indicate inactive state: {}",
                text
            );
        },
        other => panic!("expected Output from wal status, got {:?}", other),
    }
}
