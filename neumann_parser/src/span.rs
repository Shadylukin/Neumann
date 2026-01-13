//! Source location tracking for error reporting.
//!
//! Provides types for tracking positions within source code:
//! - `BytePos`: A byte offset into source text
//! - `Span`: A range of bytes representing a source region
//! - `Spanned<T>`: A value paired with its source location

use std::{fmt, ops::Range};

/// A byte position in source code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BytePos(pub u32);

impl BytePos {
    /// Creates a new byte position.
    #[inline]
    pub const fn new(pos: u32) -> Self {
        Self(pos)
    }

    /// Returns the byte offset as usize.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Advances the position by `n` bytes.
    #[inline]
    pub const fn advance(self, n: u32) -> Self {
        Self(self.0 + n)
    }
}

impl From<u32> for BytePos {
    fn from(pos: u32) -> Self {
        Self(pos)
    }
}

impl From<usize> for BytePos {
    fn from(pos: usize) -> Self {
        Self(pos as u32)
    }
}

impl fmt::Display for BytePos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A span representing a range of source code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    /// Start position (inclusive).
    pub start: BytePos,
    /// End position (exclusive).
    pub end: BytePos,
}

impl Span {
    /// Creates a new span from start to end positions.
    #[inline]
    pub const fn new(start: BytePos, end: BytePos) -> Self {
        Self { start, end }
    }

    /// Creates a span from byte offsets.
    #[inline]
    pub const fn from_offsets(start: u32, end: u32) -> Self {
        Self {
            start: BytePos(start),
            end: BytePos(end),
        }
    }

    /// Creates a zero-width span at a position.
    #[inline]
    pub const fn point(pos: BytePos) -> Self {
        Self {
            start: pos,
            end: pos,
        }
    }

    /// Creates a dummy/unknown span.
    #[inline]
    pub const fn dummy() -> Self {
        Self {
            start: BytePos(0),
            end: BytePos(0),
        }
    }

    /// Returns the length in bytes.
    #[inline]
    pub const fn len(&self) -> u32 {
        self.end.0 - self.start.0
    }

    /// Returns true if the span has zero length.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start.0 == self.end.0
    }

    /// Returns true if this is a dummy span.
    #[inline]
    pub const fn is_dummy(&self) -> bool {
        self.start.0 == 0 && self.end.0 == 0
    }

    /// Combines two spans into one that covers both.
    #[inline]
    pub const fn merge(self, other: Span) -> Span {
        let start = if self.start.0 < other.start.0 {
            self.start
        } else {
            other.start
        };
        let end = if self.end.0 > other.end.0 {
            self.end
        } else {
            other.end
        };
        Span { start, end }
    }

    /// Returns true if this span contains the given position.
    #[inline]
    pub const fn contains(&self, pos: BytePos) -> bool {
        pos.0 >= self.start.0 && pos.0 < self.end.0
    }

    /// Returns true if this span overlaps with another.
    #[inline]
    pub const fn overlaps(&self, other: Span) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }

    /// Converts to a `Range<usize>` for slicing.
    #[inline]
    pub const fn as_range(&self) -> Range<usize> {
        self.start.as_usize()..self.end.as_usize()
    }

    /// Extracts the spanned text from source.
    #[inline]
    pub fn extract<'a>(&self, source: &'a str) -> &'a str {
        &source[self.as_range()]
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start.0, self.end.0)
    }
}

impl From<Range<u32>> for Span {
    fn from(range: Range<u32>) -> Self {
        Self::from_offsets(range.start, range.end)
    }
}

impl From<Range<usize>> for Span {
    fn from(range: Range<usize>) -> Self {
        Self::from_offsets(range.start as u32, range.end as u32)
    }
}

/// A value with an associated source span.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Creates a new spanned value.
    #[inline]
    pub const fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }

    /// Maps the inner value while preserving the span.
    #[inline]
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U> {
        Spanned {
            node: f(self.node),
            span: self.span,
        }
    }

    /// Returns a reference to the inner value.
    #[inline]
    pub const fn as_ref(&self) -> Spanned<&T> {
        Spanned {
            node: &self.node,
            span: self.span,
        }
    }
}

impl<T: Copy> Copy for Spanned<T> {}

impl<T: Default> Default for Spanned<T> {
    fn default() -> Self {
        Self {
            node: T::default(),
            span: Span::dummy(),
        }
    }
}

/// Computes line and column from a byte position.
pub fn line_col(source: &str, pos: BytePos) -> (usize, usize) {
    let offset = pos.as_usize().min(source.len());
    let mut line = 1;
    let mut col = 1;

    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Returns the line containing a position.
pub fn get_line(source: &str, pos: BytePos) -> &str {
    let offset = pos.as_usize().min(source.len());

    let line_start = source[..offset].rfind('\n').map(|i| i + 1).unwrap_or(0);

    let line_end = source[offset..]
        .find('\n')
        .map(|i| offset + i)
        .unwrap_or(source.len());

    &source[line_start..line_end]
}

/// Returns the line number (1-indexed) for a position.
pub fn line_number(source: &str, pos: BytePos) -> usize {
    let offset = pos.as_usize().min(source.len());
    source[..offset].matches('\n').count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_pos() {
        let pos = BytePos::new(42);
        assert_eq!(pos.as_usize(), 42);
        assert_eq!(pos.advance(10), BytePos::new(52));

        let pos2: BytePos = 100u32.into();
        assert_eq!(pos2.0, 100);

        let pos3: BytePos = 50usize.into();
        assert_eq!(pos3.0, 50);
    }

    #[test]
    fn test_span_creation() {
        let span = Span::new(BytePos(10), BytePos(20));
        assert_eq!(span.start, BytePos(10));
        assert_eq!(span.end, BytePos(20));
        assert_eq!(span.len(), 10);
        assert!(!span.is_empty());

        let point = Span::point(BytePos(5));
        assert!(point.is_empty());

        let dummy = Span::dummy();
        assert!(dummy.is_dummy());

        let from_offsets = Span::from_offsets(5, 15);
        assert_eq!(from_offsets.len(), 10);
    }

    #[test]
    fn test_span_merge() {
        let a = Span::from_offsets(10, 20);
        let b = Span::from_offsets(15, 30);
        let merged = a.merge(b);
        assert_eq!(merged.start, BytePos(10));
        assert_eq!(merged.end, BytePos(30));

        let c = Span::from_offsets(5, 8);
        let merged2 = a.merge(c);
        assert_eq!(merged2.start, BytePos(5));
        assert_eq!(merged2.end, BytePos(20));
    }

    #[test]
    fn test_span_contains() {
        let span = Span::from_offsets(10, 20);
        assert!(span.contains(BytePos(10)));
        assert!(span.contains(BytePos(15)));
        assert!(!span.contains(BytePos(20)));
        assert!(!span.contains(BytePos(5)));
    }

    #[test]
    fn test_span_overlaps() {
        let a = Span::from_offsets(10, 20);
        let b = Span::from_offsets(15, 25);
        let c = Span::from_offsets(20, 30);
        let d = Span::from_offsets(5, 15);

        assert!(a.overlaps(b));
        assert!(!a.overlaps(c));
        assert!(a.overlaps(d));
    }

    #[test]
    fn test_span_extract() {
        let source = "SELECT * FROM users";
        let span = Span::from_offsets(7, 8);
        assert_eq!(span.extract(source), "*");

        let span2 = Span::from_offsets(14, 19);
        assert_eq!(span2.extract(source), "users");
    }

    #[test]
    fn test_span_from_range() {
        let span: Span = (5u32..10u32).into();
        assert_eq!(span.start, BytePos(5));
        assert_eq!(span.end, BytePos(10));

        let span2: Span = (5usize..10usize).into();
        assert_eq!(span2.start, BytePos(5));
        assert_eq!(span2.end, BytePos(10));
    }

    #[test]
    fn test_spanned() {
        let spanned = Spanned::new(42, Span::from_offsets(0, 2));
        assert_eq!(spanned.node, 42);
        assert_eq!(spanned.span.len(), 2);

        let mapped = spanned.map(|n| n * 2);
        assert_eq!(mapped.node, 84);
        assert_eq!(mapped.span, spanned.span);

        let ref_spanned = mapped.as_ref();
        assert_eq!(*ref_spanned.node, 84);
    }

    #[test]
    fn test_line_col() {
        let source = "line1\nline2\nline3";

        assert_eq!(line_col(source, BytePos(0)), (1, 1));
        assert_eq!(line_col(source, BytePos(5)), (1, 6));
        assert_eq!(line_col(source, BytePos(6)), (2, 1));
        assert_eq!(line_col(source, BytePos(10)), (2, 5));
        assert_eq!(line_col(source, BytePos(12)), (3, 1));
    }

    #[test]
    fn test_get_line() {
        let source = "line1\nline2\nline3";

        assert_eq!(get_line(source, BytePos(0)), "line1");
        assert_eq!(get_line(source, BytePos(3)), "line1");
        assert_eq!(get_line(source, BytePos(6)), "line2");
        assert_eq!(get_line(source, BytePos(12)), "line3");
    }

    #[test]
    fn test_line_number() {
        let source = "line1\nline2\nline3";

        assert_eq!(line_number(source, BytePos(0)), 1);
        assert_eq!(line_number(source, BytePos(5)), 1);
        assert_eq!(line_number(source, BytePos(6)), 2);
        assert_eq!(line_number(source, BytePos(12)), 3);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BytePos(42)), "42");
        assert_eq!(format!("{}", Span::from_offsets(10, 20)), "10..20");
    }

    #[test]
    fn test_default() {
        let pos: BytePos = Default::default();
        assert_eq!(pos, BytePos(0));

        let span: Span = Default::default();
        assert_eq!(span, Span::dummy());

        let spanned: Spanned<i32> = Default::default();
        assert_eq!(spanned.node, 0);
        assert!(spanned.span.is_dummy());
    }

    #[test]
    fn test_as_range() {
        let span = Span::from_offsets(5, 15);
        assert_eq!(span.as_range(), 5..15);
    }

    #[test]
    fn test_edge_cases() {
        // Position beyond source length
        let source = "short";
        assert_eq!(line_col(source, BytePos(100)), (1, 6));
        assert_eq!(line_number(source, BytePos(100)), 1);
        assert_eq!(get_line(source, BytePos(100)), "short");

        // Empty source
        let empty = "";
        assert_eq!(line_col(empty, BytePos(0)), (1, 1));
        assert_eq!(line_number(empty, BytePos(0)), 1);
        assert_eq!(get_line(empty, BytePos(0)), "");
    }
}
