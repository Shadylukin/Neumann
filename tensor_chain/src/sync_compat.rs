// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Synchronization compatibility for loom testing.
//!
//! Re-exports `parking_lot::RwLock` and `parking_lot::Mutex` in normal builds
//! and provides thin wrappers around `loom::sync` equivalents when the `loom`
//! feature is active. The wrappers strip the `Result` returns from loom's API
//! so call sites stay identical.

#[cfg(not(feature = "loom"))]
pub use parking_lot::{Mutex, RwLock};

#[cfg(feature = "loom")]
pub use self::loom_compat::{Mutex, RwLock};

#[cfg(feature = "loom")]
mod loom_compat {
    use std::fmt;

    /// Thin wrapper matching `parking_lot::RwLock` API (no `Result` from read/write).
    pub struct RwLock<T>(loom::sync::RwLock<T>);

    impl<T> RwLock<T> {
        pub fn new(t: T) -> Self {
            Self(loom::sync::RwLock::new(t))
        }

        pub fn read(&self) -> loom::sync::RwLockReadGuard<'_, T> {
            self.0.read().unwrap()
        }

        pub fn write(&self) -> loom::sync::RwLockWriteGuard<'_, T> {
            self.0.write().unwrap()
        }
    }

    impl<T: Default> Default for RwLock<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    impl<T: fmt::Debug> fmt::Debug for RwLock<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0.try_read() {
                Ok(guard) => f.debug_tuple("RwLock").field(&*guard).finish(),
                Err(_) => f.debug_tuple("RwLock").field(&"<locked>").finish(),
            }
        }
    }

    /// Thin wrapper matching `parking_lot::Mutex` API (no `Result` from lock).
    pub struct Mutex<T>(loom::sync::Mutex<T>);

    impl<T> Mutex<T> {
        pub fn new(t: T) -> Self {
            Self(loom::sync::Mutex::new(t))
        }

        pub fn lock(&self) -> loom::sync::MutexGuard<'_, T> {
            self.0.lock().unwrap()
        }
    }

    impl<T: Default> Default for Mutex<T> {
        fn default() -> Self {
            Self::new(T::default())
        }
    }

    impl<T: fmt::Debug> fmt::Debug for Mutex<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0.try_lock() {
                Ok(guard) => f.debug_tuple("Mutex").field(&*guard).finish(),
                Err(_) => f.debug_tuple("Mutex").field(&"<locked>").finish(),
            }
        }
    }
}
