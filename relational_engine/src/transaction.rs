//! Transaction support for `RelationalEngine`.
//!
//! Provides ACID transactions with row-level locking and undo logging:
//! - Begin/commit/rollback API
//! - Row-level locks with deadlock timeout
//! - Undo log for automatic rollback
//!
//! # Transaction Lifecycle
//!
//! | Phase | Description | Transitions |
//! |-------|-------------|-------------|
//! | Active | Operations allowed | Committing, Aborting |
//! | Committing | Finalizing changes | Committed |
//! | Committed | Changes permanent | (terminal) |
//! | Aborting | Rolling back | Aborted |
//! | Aborted | Changes reverted | (terminal) |
//!
//! # Lock Semantics
//!
//! Row locks are acquired on first write and held until commit/rollback:
//! - Locks have configurable timeout (default: no timeout)
//! - Expired locks are automatically cleaned up
//! - Lock conflicts return `RelationalError::LockConflict`

use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::{SlabColumnValue, SlabRowId, Value};

/// Transaction phase state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxPhase {
    /// Operations allowed.
    Active,
    /// Commit in progress.
    Committing,
    /// Successfully committed.
    Committed,
    /// Rollback in progress.
    Aborting,
    /// Rolled back.
    Aborted,
}

/// Index change record for rollback.
#[derive(Debug, Clone)]
pub struct IndexChange {
    /// Column name.
    pub column: String,
    /// Old value.
    pub old_value: Value,
    /// New value.
    pub new_value: Value,
}

/// Undo entry for rollback.
#[derive(Debug, Clone)]
pub enum UndoEntry {
    /// Undo an insert: delete the row.
    InsertedRow {
        /// Table name.
        table: String,
        /// Row ID in the slab.
        slab_row_id: SlabRowId,
        /// Engine row ID.
        row_id: u64,
        /// Index entries to remove on rollback.
        index_entries: Vec<(String, Value)>,
    },
    /// Undo an update: restore old values.
    UpdatedRow {
        /// Table name.
        table: String,
        /// Row ID in the slab.
        slab_row_id: SlabRowId,
        /// Engine row ID.
        row_id: u64,
        /// Old column values.
        old_values: Vec<SlabColumnValue>,
        /// Index changes to revert.
        index_changes: Vec<IndexChange>,
    },
    /// Undo a delete: restore the row.
    DeletedRow {
        /// Table name.
        table: String,
        /// Row ID in the slab.
        slab_row_id: SlabRowId,
        /// Engine row ID.
        row_id: u64,
        /// Old column values.
        old_values: Vec<SlabColumnValue>,
        /// Index entries to restore.
        index_entries: Vec<(String, Value)>,
    },
}

/// Active transaction state.
#[derive(Debug)]
pub struct Transaction {
    /// Transaction ID.
    pub tx_id: u64,
    /// Current phase.
    pub phase: TxPhase,
    /// When the transaction started (epoch milliseconds).
    pub started_at_ms: u64,
    /// Transaction timeout in milliseconds.
    pub timeout_ms: u64,
    /// Undo log for rollback.
    pub undo_log: Vec<UndoEntry>,
    /// Tables affected by this transaction.
    pub affected_tables: HashSet<String>,
}

impl Transaction {
    /// Creates a new transaction.
    #[must_use]
    pub fn new(tx_id: u64, timeout_ms: u64) -> Self {
        Self {
            tx_id,
            phase: TxPhase::Active,
            started_at_ms: now_epoch_millis(),
            timeout_ms,
            undo_log: Vec::new(),
            affected_tables: HashSet::new(),
        }
    }

    /// Returns true if the transaction is active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.phase == TxPhase::Active
    }

    /// Returns true if the transaction has timed out.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        now_epoch_millis().saturating_sub(self.started_at_ms) > self.timeout_ms
    }

    /// Records an undo entry for rollback.
    pub fn record_undo(&mut self, entry: UndoEntry) {
        match &entry {
            UndoEntry::InsertedRow { table, .. }
            | UndoEntry::UpdatedRow { table, .. }
            | UndoEntry::DeletedRow { table, .. } => {
                self.affected_tables.insert(table.clone());
            },
        }
        self.undo_log.push(entry);
    }
}

/// Row-level lock for transactions.
#[derive(Debug, Clone)]
pub struct RowLock {
    /// Table name.
    pub table: String,
    /// Row ID (1-based engine row ID).
    pub row_id: u64,
    /// Transaction ID holding the lock.
    pub tx_id: u64,
    /// When the lock was acquired (epoch milliseconds).
    pub acquired_at_ms: u64,
    /// Lock timeout in milliseconds.
    pub timeout_ms: u64,
}

impl RowLock {
    /// Returns true if this lock has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        now_epoch_millis().saturating_sub(self.acquired_at_ms) > self.timeout_ms
    }
}

/// Monotonic deadline for query timeout checking.
///
/// Uses `Instant` for monotonic time that is immune to system clock changes.
/// This is essential for query timeouts where wall-clock adjustments should
/// not affect timeout behavior.
#[derive(Debug, Clone, Copy)]
pub struct Deadline {
    deadline: Option<Instant>,
}

impl Deadline {
    /// Creates a deadline from a timeout in milliseconds.
    ///
    /// Returns a deadline that will expire after the specified duration.
    /// If `timeout_ms` is `None`, the deadline will never expire.
    #[must_use]
    pub fn from_timeout_ms(timeout_ms: Option<u64>) -> Self {
        Self {
            deadline: timeout_ms.map(|ms| Instant::now() + Duration::from_millis(ms)),
        }
    }

    /// Creates a deadline that never expires.
    #[must_use]
    pub const fn never() -> Self {
        Self { deadline: None }
    }

    /// Returns true if the deadline has expired.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// Returns the remaining time in milliseconds, or `None` if no deadline is set.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn remaining_ms(&self) -> Option<u64> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()).as_millis() as u64)
    }
}

impl Default for Deadline {
    fn default() -> Self {
        Self::never()
    }
}

/// Row-level lock manager for transactions.
#[derive(Debug)]
pub struct RowLockManager {
    /// Locks by (table, `row_id`) -> lock.
    locks: RwLock<HashMap<(String, u64), RowLock>>,
    /// Locks by `tx_id` -> list of (table, `row_id`).
    tx_locks: RwLock<HashMap<u64, Vec<(String, u64)>>>,
    /// Default lock timeout.
    pub default_timeout: Duration,
}

impl Default for RowLockManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RowLockManager {
    /// Creates a new lock manager with 30 second default timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            tx_locks: RwLock::new(HashMap::new()),
            default_timeout: Duration::from_secs(30),
        }
    }

    /// Creates a lock manager with custom default timeout.
    #[must_use]
    pub fn with_default_timeout(timeout: Duration) -> Self {
        Self {
            locks: RwLock::new(HashMap::new()),
            tx_locks: RwLock::new(HashMap::new()),
            default_timeout: timeout,
        }
    }

    /// Try to acquire locks for a set of rows.
    ///
    /// # Errors
    ///
    /// Returns `LockConflictInfo` if a row is already locked by another transaction.
    #[allow(clippy::significant_drop_tightening)]
    pub fn try_lock(
        &self,
        tx_id: u64,
        rows: &[(String, u64)],
    ) -> std::result::Result<(), LockConflictInfo> {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        // First check if all rows are available
        for (table, row_id) in rows {
            let key = (table.clone(), *row_id);
            if let Some(existing) = locks.get(&key) {
                if !existing.is_expired() && existing.tx_id != tx_id {
                    return Err(LockConflictInfo {
                        blocking_tx: existing.tx_id,
                        table: table.clone(),
                        row_id: *row_id,
                    });
                }
            }
        }

        // Acquire all locks
        let now_ms = now_epoch_millis();
        #[allow(clippy::cast_possible_truncation)]
        let timeout_ms = self.default_timeout.as_millis() as u64;

        for (table, row_id) in rows {
            let key = (table.clone(), *row_id);
            locks.insert(
                key.clone(),
                RowLock {
                    table: table.clone(),
                    row_id: *row_id,
                    tx_id,
                    acquired_at_ms: now_ms,
                    timeout_ms,
                },
            );
            tx_locks.entry(tx_id).or_default().push(key);
        }

        Ok(())
    }

    /// Release all locks held by a transaction.
    pub fn release(&self, tx_id: u64) {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        if let Some(keys) = tx_locks.remove(&tx_id) {
            for key in keys {
                if let Some(lock) = locks.get(&key) {
                    if lock.tx_id == tx_id {
                        locks.remove(&key);
                    }
                }
            }
        }
    }

    /// Check if a row is locked.
    #[must_use]
    pub fn is_locked(&self, table: &str, row_id: u64) -> bool {
        let locks = self.locks.read();
        let key = (table.to_string(), row_id);
        locks.get(&key).is_some_and(|lock| !lock.is_expired())
    }

    /// Get the transaction ID holding a lock on a row.
    #[must_use]
    pub fn lock_holder(&self, table: &str, row_id: u64) -> Option<u64> {
        let locks = self.locks.read();
        let key = (table.to_string(), row_id);
        locks
            .get(&key)
            .filter(|lock| !lock.is_expired())
            .map(|lock| lock.tx_id)
    }

    /// Clean up expired locks.
    #[allow(clippy::significant_drop_tightening)]
    pub fn cleanup_expired(&self) -> usize {
        let mut locks = self.locks.write();
        let mut tx_locks = self.tx_locks.write();

        let expired: Vec<_> = locks
            .iter()
            .filter(|(_, lock)| lock.is_expired())
            .map(|(k, lock)| (k.clone(), lock.tx_id))
            .collect();

        for (key, tx_id) in &expired {
            locks.remove(key);
            if let Some(tx_keys) = tx_locks.get_mut(tx_id) {
                tx_keys.retain(|k| k != key);
            }
        }

        expired.len()
    }

    /// Get the number of active locks.
    #[must_use]
    pub fn active_lock_count(&self) -> usize {
        self.locks.read().len()
    }

    /// Get the number of locks held by a transaction.
    #[must_use]
    pub fn locks_held_by(&self, tx_id: u64) -> usize {
        self.tx_locks.read().get(&tx_id).map_or(0, Vec::len)
    }
}

/// Information about a lock conflict.
#[derive(Debug, Clone)]
pub struct LockConflictInfo {
    /// Transaction holding the conflicting lock.
    pub blocking_tx: u64,
    /// Table name.
    pub table: String,
    /// Row ID.
    pub row_id: u64,
}

/// Transaction ID counter.
static TX_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Transaction manager for `RelationalEngine`.
#[derive(Debug)]
pub struct TransactionManager {
    /// Active transactions by ID.
    transactions: DashMap<u64, Transaction>,
    /// Row lock manager.
    lock_manager: RowLockManager,
    /// Default transaction timeout.
    pub default_timeout: Duration,
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TransactionManager {
    /// Creates a new transaction manager with 60 second default timeout.
    #[must_use]
    pub fn new() -> Self {
        Self {
            transactions: DashMap::new(),
            lock_manager: RowLockManager::new(),
            default_timeout: Duration::from_secs(60),
        }
    }

    /// Creates a transaction manager with custom timeout.
    #[must_use]
    pub fn with_timeout(timeout: Duration) -> Self {
        Self {
            transactions: DashMap::new(),
            lock_manager: RowLockManager::with_default_timeout(timeout),
            default_timeout: timeout,
        }
    }

    /// Begin a new transaction.
    #[allow(clippy::cast_possible_truncation)]
    pub fn begin(&self) -> u64 {
        let tx_id = TX_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tx = Transaction::new(tx_id, self.default_timeout.as_millis() as u64);
        self.transactions.insert(tx_id, tx);
        tx_id
    }

    /// Get a transaction by ID (read-only).
    #[must_use]
    pub fn get(&self, tx_id: u64) -> Option<TxPhase> {
        self.transactions.get(&tx_id).map(|r| r.phase)
    }

    /// Check if a transaction is active.
    #[must_use]
    pub fn is_active(&self, tx_id: u64) -> bool {
        self.transactions.get(&tx_id).is_some_and(|r| r.is_active())
    }

    /// Set transaction phase.
    pub fn set_phase(&self, tx_id: u64, phase: TxPhase) -> bool {
        if let Some(mut tx) = self.transactions.get_mut(&tx_id) {
            tx.phase = phase;
            true
        } else {
            false
        }
    }

    /// Record an undo entry for a transaction.
    #[allow(clippy::option_if_let_else)]
    pub fn record_undo(&self, tx_id: u64, entry: UndoEntry) -> bool {
        if let Some(mut tx) = self.transactions.get_mut(&tx_id) {
            tx.record_undo(entry);
            true
        } else {
            false
        }
    }

    /// Get the undo log for a transaction (cloned for rollback).
    #[must_use]
    pub fn get_undo_log(&self, tx_id: u64) -> Option<Vec<UndoEntry>> {
        self.transactions.get(&tx_id).map(|r| r.undo_log.clone())
    }

    /// Remove a completed transaction.
    pub fn remove(&self, tx_id: u64) {
        self.transactions.remove(&tx_id);
    }

    /// Get the lock manager.
    #[must_use]
    pub const fn lock_manager(&self) -> &RowLockManager {
        &self.lock_manager
    }

    /// Release all locks for a transaction.
    pub fn release_locks(&self, tx_id: u64) {
        self.lock_manager.release(tx_id);
    }

    /// Get number of active transactions.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.transactions.iter().filter(|r| r.is_active()).count()
    }

    /// Clean up expired transactions.
    pub fn cleanup_expired(&self) -> usize {
        let before = self.transactions.len();
        self.transactions.retain(|_, tx| {
            if tx.is_expired() {
                self.lock_manager.release(tx.tx_id);
                false
            } else {
                true
            }
        });
        before - self.transactions.len()
    }
}

/// Get current epoch time in milliseconds.
///
/// # Panics
///
/// Panics if system clock is before UNIX epoch (1970-01-01), indicating
/// a misconfigured system clock.
#[allow(clippy::cast_possible_truncation)]
fn now_epoch_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is before UNIX epoch (1970-01-01)")
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tx_phase_transitions() {
        let tx = Transaction::new(1, 60000);
        assert_eq!(tx.phase, TxPhase::Active);
        assert!(tx.is_active());
    }

    #[test]
    fn test_tx_expiration() {
        let mut tx = Transaction::new(1, 0); // Immediate expiration
        tx.started_at_ms = 0; // Force old timestamp
        assert!(tx.is_expired());
    }

    #[test]
    fn test_tx_record_undo() {
        let mut tx = Transaction::new(1, 60000);
        tx.record_undo(UndoEntry::InsertedRow {
            table: "users".to_string(),
            slab_row_id: SlabRowId::new(0),
            row_id: 1,
            index_entries: vec![],
        });
        assert_eq!(tx.undo_log.len(), 1);
        assert!(tx.affected_tables.contains("users"));
    }

    #[test]
    fn test_row_lock_manager_basic() {
        let mgr = RowLockManager::new();

        // Lock row
        let result = mgr.try_lock(1, &[("users".to_string(), 1)]);
        assert!(result.is_ok());
        assert!(mgr.is_locked("users", 1));
        assert_eq!(mgr.lock_holder("users", 1), Some(1));

        // Conflict with different tx
        let result = mgr.try_lock(2, &[("users".to_string(), 1)]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.blocking_tx, 1);

        // Same tx can re-lock
        let result = mgr.try_lock(1, &[("users".to_string(), 1)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_row_lock_manager_release() {
        let mgr = RowLockManager::new();

        mgr.try_lock(1, &[("users".to_string(), 1), ("users".to_string(), 2)])
            .unwrap();
        assert_eq!(mgr.locks_held_by(1), 2);

        mgr.release(1);
        assert!(!mgr.is_locked("users", 1));
        assert!(!mgr.is_locked("users", 2));
        assert_eq!(mgr.locks_held_by(1), 0);
    }

    #[test]
    fn test_transaction_manager_lifecycle() {
        let mgr = TransactionManager::new();

        let tx_id = mgr.begin();
        assert!(mgr.is_active(tx_id));
        assert_eq!(mgr.get(tx_id), Some(TxPhase::Active));

        mgr.set_phase(tx_id, TxPhase::Committing);
        assert_eq!(mgr.get(tx_id), Some(TxPhase::Committing));
        assert!(!mgr.is_active(tx_id));

        mgr.set_phase(tx_id, TxPhase::Committed);
        mgr.remove(tx_id);
        assert_eq!(mgr.get(tx_id), None);
    }

    #[test]
    fn test_transaction_manager_undo_log() {
        let mgr = TransactionManager::new();
        let tx_id = mgr.begin();

        mgr.record_undo(
            tx_id,
            UndoEntry::InsertedRow {
                table: "users".to_string(),
                slab_row_id: SlabRowId::new(0),
                row_id: 1,
                index_entries: vec![],
            },
        );

        let undo_log = mgr.get_undo_log(tx_id).unwrap();
        assert_eq!(undo_log.len(), 1);
    }

    #[test]
    fn test_lock_conflict_info() {
        let info = LockConflictInfo {
            blocking_tx: 1,
            table: "users".to_string(),
            row_id: 42,
        };
        assert_eq!(info.blocking_tx, 1);
        assert_eq!(info.table, "users");
        assert_eq!(info.row_id, 42);
    }

    #[test]
    fn test_row_lock_expiration() {
        let lock = RowLock {
            table: "users".to_string(),
            row_id: 1,
            tx_id: 1,
            acquired_at_ms: 0, // Old timestamp
            timeout_ms: 1,     // 1ms timeout
        };
        assert!(lock.is_expired());
    }

    #[test]
    fn test_row_lock_manager_multiple_rows() {
        let mgr = RowLockManager::new();

        let rows = vec![
            ("users".to_string(), 1),
            ("users".to_string(), 2),
            ("orders".to_string(), 1),
        ];

        let result = mgr.try_lock(1, &rows);
        assert!(result.is_ok());

        assert!(mgr.is_locked("users", 1));
        assert!(mgr.is_locked("users", 2));
        assert!(mgr.is_locked("orders", 1));
        assert!(!mgr.is_locked("users", 3));
    }

    #[test]
    fn test_transaction_manager_active_count() {
        let mgr = TransactionManager::new();

        let tx1 = mgr.begin();
        let tx2 = mgr.begin();
        assert_eq!(mgr.active_count(), 2);

        mgr.set_phase(tx1, TxPhase::Committed);
        assert_eq!(mgr.active_count(), 1);

        mgr.set_phase(tx2, TxPhase::Aborted);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_undo_entry_variants() {
        let insert = UndoEntry::InsertedRow {
            table: "t".to_string(),
            slab_row_id: SlabRowId::new(0),
            row_id: 1,
            index_entries: vec![("col".to_string(), Value::Int(42))],
        };

        let update = UndoEntry::UpdatedRow {
            table: "t".to_string(),
            slab_row_id: SlabRowId::new(0),
            row_id: 1,
            old_values: vec![SlabColumnValue::Int(42)],
            index_changes: vec![IndexChange {
                column: "col".to_string(),
                old_value: Value::Int(42),
                new_value: Value::Int(43),
            }],
        };

        let delete = UndoEntry::DeletedRow {
            table: "t".to_string(),
            slab_row_id: SlabRowId::new(0),
            row_id: 1,
            old_values: vec![SlabColumnValue::Int(42)],
            index_entries: vec![("col".to_string(), Value::Int(42))],
        };

        // Verify enum variants compile and match
        match insert {
            UndoEntry::InsertedRow { row_id, .. } => assert_eq!(row_id, 1),
            _ => panic!("Wrong variant"),
        }

        match update {
            UndoEntry::UpdatedRow { old_values, .. } => assert_eq!(old_values.len(), 1),
            _ => panic!("Wrong variant"),
        }

        match delete {
            UndoEntry::DeletedRow { index_entries, .. } => assert_eq!(index_entries.len(), 1),
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_index_change() {
        let change = IndexChange {
            column: "age".to_string(),
            old_value: Value::Int(30),
            new_value: Value::Int(31),
        };
        assert_eq!(change.column, "age");
    }

    #[test]
    fn test_lock_manager_default() {
        let mgr = RowLockManager::default();
        assert_eq!(mgr.active_lock_count(), 0);
    }

    #[test]
    fn test_transaction_manager_default() {
        let mgr = TransactionManager::default();
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_cleanup_expired_locks() {
        let mgr = RowLockManager::with_default_timeout(Duration::from_millis(1));

        mgr.try_lock(1, &[("users".to_string(), 1)]).unwrap();

        // Wait for lock to expire
        std::thread::sleep(Duration::from_millis(5));

        let cleaned = mgr.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert!(!mgr.is_locked("users", 1));
    }

    #[test]
    fn test_now_epoch_millis_returns_reasonable_value() {
        let now = now_epoch_millis();
        // Should be after 2020-01-01 (1577836800000 ms)
        assert!(now > 1_577_836_800_000, "epoch time should be after 2020");
        // Should be before 2100-01-01 (4102444800000 ms)
        assert!(now < 4_102_444_800_000, "epoch time should be before 2100");
    }

    #[test]
    fn test_deadline_from_timeout_ms() {
        let deadline = Deadline::from_timeout_ms(Some(1000));
        assert!(!deadline.is_expired());
        assert!(deadline.remaining_ms().is_some());
    }

    #[test]
    fn test_deadline_never_expires() {
        let deadline = Deadline::never();
        assert!(!deadline.is_expired());
        assert!(deadline.remaining_ms().is_none());
    }

    #[test]
    fn test_deadline_is_expired() {
        // Create a deadline that should already be expired
        let deadline = Deadline::from_timeout_ms(Some(0));
        // Give it a tiny bit of time to pass
        std::thread::sleep(Duration::from_millis(1));
        assert!(deadline.is_expired());
    }

    #[test]
    fn test_deadline_default() {
        let deadline = Deadline::default();
        assert!(!deadline.is_expired());
        assert!(deadline.remaining_ms().is_none());
    }

    #[test]
    fn test_deadline_none_timeout() {
        let deadline = Deadline::from_timeout_ms(None);
        assert!(!deadline.is_expired());
        assert!(deadline.remaining_ms().is_none());
    }

    #[test]
    fn test_zero_timeout_immediate_expiry() {
        let deadline = Deadline::from_timeout_ms(Some(0));
        // Wait a tiny bit to ensure the deadline passes
        std::thread::sleep(Duration::from_millis(1));
        assert!(deadline.is_expired());
    }
}
