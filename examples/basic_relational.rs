// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Basic relational database operations example.
//!
//! This example demonstrates:
//! - Creating tables with schemas
//! - Inserting rows
//! - Querying with conditions
//! - Updating and deleting rows
//!
//! Run with: `cargo run --example basic_relational`

use relational_engine::{Column, ColumnType, Condition, RelationalEngine, Schema, Value};

fn main() {
    println!("Neumann Relational Engine Example\n");

    // Create a new relational engine
    let engine = RelationalEngine::new();

    // Define a schema for a users table
    let schema = Schema::new(vec![
        Column::new("id", ColumnType::Int),
        Column::new("name", ColumnType::String),
        Column::new("email", ColumnType::String),
        Column::new("age", ColumnType::Int),
        Column::new("active", ColumnType::Bool),
    ]);

    // Create the table
    engine
        .create_table("users", schema)
        .expect("Failed to create table");
    println!("Created 'users' table");

    // Insert some rows
    let users = vec![
        vec![
            ("id", Value::Int(1)),
            ("name", Value::String("Alice".to_string())),
            ("email", Value::String("alice@example.com".to_string())),
            ("age", Value::Int(30)),
            ("active", Value::Bool(true)),
        ],
        vec![
            ("id", Value::Int(2)),
            ("name", Value::String("Bob".to_string())),
            ("email", Value::String("bob@example.com".to_string())),
            ("age", Value::Int(25)),
            ("active", Value::Bool(true)),
        ],
        vec![
            ("id", Value::Int(3)),
            ("name", Value::String("Charlie".to_string())),
            ("email", Value::String("charlie@example.com".to_string())),
            ("age", Value::Int(35)),
            ("active", Value::Bool(false)),
        ],
    ];

    for user in users {
        let values: std::collections::HashMap<String, Value> =
            user.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        engine.insert("users", values).expect("Failed to insert");
    }
    println!("Inserted 3 users\n");

    // Query all users
    println!("All users:");
    let all_users = engine
        .select("users", Condition::True)
        .expect("Failed to select");
    for row in &all_users {
        println!("  {:?}", row);
    }
    println!();

    // Query users with a condition (age > 28)
    println!("Users with age > 28:");
    let older_users = engine
        .select("users", Condition::Gt("age".to_string(), Value::Int(28)))
        .expect("Failed to select");
    for row in &older_users {
        println!("  {:?}", row);
    }
    println!();

    // Query active users
    println!("Active users:");
    let active_users = engine
        .select(
            "users",
            Condition::Eq("active".to_string(), Value::Bool(true)),
        )
        .expect("Failed to select");
    for row in &active_users {
        println!("  {:?}", row);
    }
    println!();

    // Update a user
    let mut updates = std::collections::HashMap::new();
    updates.insert("active".to_string(), Value::Bool(true));
    let updated = engine
        .update(
            "users",
            Condition::Eq("name".to_string(), Value::String("Charlie".to_string())),
            updates,
        )
        .expect("Failed to update");
    println!("Updated {} row(s) - Charlie is now active\n", updated);

    // Delete a user
    let deleted = engine
        .delete_rows("users", Condition::Eq("id".to_string(), Value::Int(2)))
        .expect("Failed to delete");
    println!("Deleted {} row(s) - Removed Bob\n", deleted);

    // Final state
    println!("Final users:");
    let final_users = engine
        .select("users", Condition::True)
        .expect("Failed to select");
    for row in &final_users {
        println!("  {:?}", row);
    }

    // Show table count
    let count = engine
        .count("users", Condition::True)
        .expect("Failed to count");
    println!("\nTotal users: {}", count);
}
