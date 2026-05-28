// SPDX-License-Identifier: MIT OR Apache-2.0
//! `VAULT` statement execution.

use neumann_parser::{VaultOp, VaultStmt};
use tensor_checkpoint::DestructiveOp;

use crate::policy::ProtectedOpResult;
use crate::{protection, QueryResult, QueryRouter, Result, RouterError};

use super::expr;

/// Execute a `VAULT ...` statement. Requires an authenticated identity.
pub fn exec_vault(router: &QueryRouter, stmt: &VaultStmt) -> Result<QueryResult> {
    let vault = router
        .vault
        .as_ref()
        .ok_or_else(|| RouterError::VaultError("Vault not initialized".to_string()))?;

    // SECURITY: Require explicit authentication for vault operations
    let identity = router.require_identity()?;

    match &stmt.operation {
        VaultOp::Set { key, value } => {
            let key_str = expr::eval_string_expr(key)?;
            let value_str = expr::eval_string_expr(value)?;
            vault.set(identity, &key_str, &value_str)?;
            Ok(QueryResult::Empty)
        },
        VaultOp::Get { key } => {
            let key_str = expr::eval_string_expr(key)?;
            let value = vault.get(identity, &key_str)?;
            Ok(QueryResult::Value(value))
        },
        VaultOp::Delete { key } => {
            let key_str = expr::eval_string_expr(key)?;

            // Check for auto-checkpoint protection (don't show secret value!)
            let op = DestructiveOp::VaultDelete {
                key: key_str.clone(),
            };

            match protection::protect_destructive_op(
                router,
                &format!("VAULT DELETE '{key_str}'"),
                op,
                vec![format!("secret key: {}", key_str)],
            ) {
                ProtectedOpResult::Proceed => {},
                ProtectedOpResult::Cancelled => {
                    return Err(RouterError::CheckpointError(
                        "Operation cancelled by user".to_string(),
                    ));
                },
            }

            vault.delete(identity, &key_str)?;
            Ok(QueryResult::Empty)
        },
        VaultOp::List { pattern } => {
            let pat = pattern
                .as_ref()
                .map(expr::eval_string_expr)
                .transpose()?
                .unwrap_or_else(|| "*".to_string());
            let keys = vault.list(identity, &pat)?;
            Ok(QueryResult::Value(keys.join("\n")))
        },
        VaultOp::Rotate { key, new_value } => {
            let key_str = expr::eval_string_expr(key)?;
            let new_value_str = expr::eval_string_expr(new_value)?;
            vault.rotate(identity, &key_str, &new_value_str)?;
            Ok(QueryResult::Empty)
        },
        VaultOp::Grant { entity, key } => {
            let entity_str = expr::eval_string_expr(entity)?;
            let key_str = expr::eval_string_expr(key)?;
            vault.grant(identity, &entity_str, &key_str)?;
            Ok(QueryResult::Empty)
        },
        VaultOp::Revoke { entity, key } => {
            let entity_str = expr::eval_string_expr(entity)?;
            let key_str = expr::eval_string_expr(key)?;
            vault.revoke(identity, &entity_str, &key_str)?;
            Ok(QueryResult::Empty)
        },
    }
}
