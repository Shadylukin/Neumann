//! Iterator over references to spatial entries.

use crate::bbox::SpatialEntryN;

/// Iterator over references to spatial entries in a `SpatialIndexN`.
pub struct SpatialIterN<'a, const D: usize, T> {
    /// Collected references to all entries.
    pub(crate) entries: Vec<&'a SpatialEntryN<D, T>>,
    /// Current iteration position.
    pub(crate) pos: usize,
}

impl<'a, const D: usize, T> Iterator for SpatialIterN<'a, D, T> {
    type Item = &'a SpatialEntryN<D, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let entry = self.entries[self.pos];
            self.pos += 1;
            Some(entry)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<const D: usize, T> ExactSizeIterator for SpatialIterN<'_, D, T> {}
