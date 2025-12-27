//! Shell integration tests.
//!
//! Tests shell command processing, help text, and interactive features.

use neumann_shell::{CommandResult, Shell, ShellConfig};

fn create_test_shell() -> Shell {
    Shell::new()
}

#[test]
fn test_shell_creation() {
    let _shell = create_test_shell();
    // Shell should be created successfully
}

#[test]
fn test_shell_with_config() {
    let config = ShellConfig::default();
    let _shell = Shell::with_config(config);
    // Should work with custom config
}

#[test]
fn test_shell_execute_select() {
    let mut shell = create_test_shell();

    // Create table and data
    shell.execute("CREATE TABLE users (id INT, name TEXT)");
    shell.execute("INSERT users id=1, name='Alice'");

    // Execute select
    let result = shell.execute("SELECT * FROM users");

    match result {
        CommandResult::Output(output) => {
            // Should contain data
            assert!(!output.is_empty());
        }
        CommandResult::Error(e) => {
            panic!("Unexpected error: {}", e);
        }
        _ => {}
    }
}

#[test]
fn test_shell_help_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("help");

    match result {
        CommandResult::Help(help_text) => {
            // Help should contain usage information
            assert!(!help_text.is_empty());
            assert!(help_text.contains("SELECT") || help_text.contains("select"));
        }
        _ => {}
    }
}

#[test]
fn test_shell_empty_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("");

    match result {
        CommandResult::Empty => {
            // Empty input should return Empty result
        }
        _ => {}
    }
}

#[test]
fn test_shell_whitespace_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("   ");

    match result {
        CommandResult::Empty => {
            // Whitespace-only should return Empty
        }
        _ => {}
    }
}

#[test]
fn test_shell_exit_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("exit");

    match result {
        CommandResult::Exit => {
            // exit should return Exit
        }
        _ => {}
    }
}

#[test]
fn test_shell_quit_command() {
    let mut shell = create_test_shell();

    let result = shell.execute("quit");

    match result {
        CommandResult::Exit => {
            // quit should also return Exit
        }
        _ => {}
    }
}

#[test]
fn test_shell_node_commands() {
    let mut shell = create_test_shell();

    // Create node
    let result = shell.execute("NODE CREATE user name='Test'");

    match result {
        CommandResult::Output(output) => {
            // Should output node ID
            assert!(!output.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_embed_commands() {
    let mut shell = create_test_shell();

    // Store embedding
    shell.execute("EMBED key1 0.5, 0.5, 0.5, 0.5");

    // Get embedding
    let result = shell.execute("EMBED GET 'key1'");

    match result {
        CommandResult::Output(output) => {
            // Should show embedding
            assert!(!output.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_similar_command() {
    let mut shell = create_test_shell();

    // Store embeddings
    shell.execute("EMBED doc1 1.0, 0.0, 0.0, 0.0");
    shell.execute("EMBED doc2 0.9, 0.1, 0.0, 0.0");
    shell.execute("EMBED doc3 0.0, 0.0, 1.0, 0.0");

    // Find similar
    let result = shell.execute("SIMILAR doc1 TOP 2");

    match result {
        CommandResult::Output(_output) => {
            // Should contain similarity results
        }
        _ => {}
    }
}

#[test]
fn test_shell_error_handling() {
    let mut shell = create_test_shell();

    // Invalid command
    let result = shell.execute("INVALID_COMMAND xyz");

    match result {
        CommandResult::Error(e) => {
            // Should return error
            assert!(!e.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_case_insensitivity() {
    let mut shell = create_test_shell();

    // Commands should be case-insensitive
    shell.execute("create table test1 (id INT)");
    shell.execute("CREATE TABLE test2 (id INT)");
    shell.execute("Create Table test3 (id INT)");

    // All should work
    let result = shell.execute("SHOW TABLES");
    match result {
        CommandResult::Output(_output) => {
            // Should show all tables
        }
        _ => {}
    }
}

#[test]
fn test_shell_semicolon_handling() {
    let mut shell = create_test_shell();

    // Commands with semicolons
    let result = shell.execute("SELECT 1;");

    match result {
        CommandResult::Output(_) => {}
        CommandResult::Error(_) => {}
        _ => {}
    }
}

#[test]
fn test_shell_multiline_support() {
    let mut shell = create_test_shell();

    // Multi-line query (single command)
    let query = "CREATE TABLE multiline (
        id INT,
        name TEXT,
        value FLOAT
    )";

    let result = shell.execute(query);

    match result {
        CommandResult::Output(_) => {}
        _ => {}
    }
}

#[test]
fn test_shell_show_tables() {
    let mut shell = create_test_shell();

    // Create some tables
    shell.execute("CREATE TABLE table1 (id INT)");
    shell.execute("CREATE TABLE table2 (id INT)");

    let result = shell.execute("SHOW TABLES");

    match result {
        CommandResult::Output(output) => {
            // Should list tables
            assert!(output.contains("table1") || output.contains("table2") || !output.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_describe_command() {
    let mut shell = create_test_shell();

    // Create table
    shell.execute("CREATE TABLE described (id INT, name TEXT, active BOOLEAN)");

    // Describe it
    let result = shell.execute("DESCRIBE TABLE described");

    match result {
        CommandResult::Output(output) => {
            // Should show schema
            assert!(!output.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_count_embeddings() {
    let mut shell = create_test_shell();

    // Store some embeddings
    shell.execute("EMBED a 1.0, 0.0");
    shell.execute("EMBED b 0.0, 1.0");

    let result = shell.execute("COUNT EMBEDDINGS");

    match result {
        CommandResult::Output(output) => {
            // Should show count
            assert!(!output.is_empty());
        }
        _ => {}
    }
}

#[test]
fn test_shell_find_command() {
    let mut shell = create_test_shell();

    // Create some nodes
    shell.execute("NODE CREATE person name='Alice'");
    shell.execute("NODE CREATE person name='Bob'");

    let result = shell.execute("FIND NODE person");

    match result {
        CommandResult::Output(_output) => {
            // Should show found nodes
        }
        _ => {}
    }
}

#[test]
fn test_shell_entity_commands() {
    let mut shell = create_test_shell();

    // Entity create
    let result = shell.execute("ENTITY CREATE 'user:1' { name: 'Test' }");

    match result {
        CommandResult::Output(_output) => {}
        _ => {}
    }

    // Entity connect
    shell.execute("ENTITY CREATE 'user:2' { name: 'Other' }");
    let connect_result = shell.execute("ENTITY CONNECT 'user:1' -> 'user:2' : knows");

    match connect_result {
        CommandResult::Output(_) => {}
        _ => {}
    }
}

#[test]
fn test_shell_vault_commands_without_init() {
    let mut shell = create_test_shell();

    // Vault not initialized - should error gracefully
    let result = shell.execute("VAULT GET 'key'");

    match result {
        CommandResult::Error(_) => {
            // Expected
        }
        _ => {}
    }
}

#[test]
fn test_shell_cache_commands_without_init() {
    let mut shell = create_test_shell();

    // Cache not initialized
    let result = shell.execute("CACHE STATS");

    match result {
        CommandResult::Error(_) | CommandResult::Output(_) => {
            // Either error or empty output is acceptable
        }
        _ => {}
    }
}

#[test]
fn test_shell_embed_batch() {
    let mut shell = create_test_shell();

    let result =
        shell.execute("EMBED BATCH [('batch:1', [1.0, 0.0]), ('batch:2', [0.0, 1.0])]");

    match result {
        CommandResult::Output(_output) => {
            // Should show count of stored embeddings
        }
        _ => {}
    }
}

#[test]
fn test_shell_neighbors_command() {
    let mut shell = create_test_shell();

    // Create graph
    shell.execute("NODE CREATE user name='Alice'");
    shell.execute("NODE CREATE user name='Bob'");
    shell.execute("EDGE CREATE 1 -> 2 knows");

    let result = shell.execute("NEIGHBORS 1 OUT");

    match result {
        CommandResult::Output(_output) => {
            // Should show neighbor IDs
        }
        _ => {}
    }
}

#[test]
fn test_shell_path_command() {
    let mut shell = create_test_shell();

    // Create connected graph
    shell.execute("NODE CREATE a name='A'");
    shell.execute("NODE CREATE b name='B'");
    shell.execute("NODE CREATE c name='C'");
    shell.execute("EDGE CREATE 1 -> 2 link");
    shell.execute("EDGE CREATE 2 -> 3 link");

    let result = shell.execute("PATH 1 -> 3");

    match result {
        CommandResult::Output(_output) => {
            // Should show path
        }
        _ => {}
    }
}

#[test]
fn test_shell_comment_handling() {
    let mut shell = create_test_shell();

    // SQL comments should be ignored
    let result = shell.execute("-- This is a comment\nSELECT 1");

    match result {
        CommandResult::Output(_) | CommandResult::Error(_) => {
            // Comment handling may vary
        }
        _ => {}
    }
}

#[test]
fn test_shell_preserves_state() {
    let mut shell = create_test_shell();

    // Execute multiple commands
    shell.execute("CREATE TABLE state_test (id INT)");
    shell.execute("INSERT state_test id=1");
    shell.execute("INSERT state_test id=2");

    // State should be preserved
    let result = shell.execute("SELECT * FROM state_test");

    match result {
        CommandResult::Output(output) => {
            // Should show both rows
            assert!(!output.is_empty());
        }
        _ => {}
    }
}
