//! Cypher-like graph query language AST types.
//!
//! Provides AST node types for Cypher-style graph queries:
//! - MATCH pattern matching
//! - CREATE node/relationship creation
//! - DELETE node/relationship deletion
//! - MERGE upsert operations

#![allow(clippy::module_name_repetitions)]

use crate::ast::{Expr, Ident, OrderByItem, Property};

/// A complete MATCH statement with all clauses.
#[derive(Clone, Debug, PartialEq)]
pub struct CypherMatchStmt {
    /// Whether this is OPTIONAL MATCH.
    pub optional: bool,
    /// Pattern(s) to match (comma-separated).
    pub patterns: Vec<CypherPattern>,
    /// WHERE clause condition.
    pub where_clause: Option<Box<Expr>>,
    /// RETURN clause projection.
    pub return_clause: CypherReturn,
    /// ORDER BY clause.
    pub order_by: Vec<OrderByItem>,
    /// SKIP clause.
    pub skip: Option<Expr>,
    /// LIMIT clause.
    pub limit: Option<Expr>,
}

/// A single pattern in a MATCH clause.
///
/// Examples:
/// - `(a:Person)-[:KNOWS]->(b:Person)`
/// - `path = (a)-[*1..5]->(b)`
#[derive(Clone, Debug, PartialEq)]
pub struct CypherPattern {
    /// Optional path variable assignment: `path = ...`
    pub variable: Option<Ident>,
    /// Pattern elements (alternating nodes and relationships).
    pub elements: Vec<CypherElement>,
}

/// A pattern element: either a node or a relationship.
#[derive(Clone, Debug, PartialEq)]
pub enum CypherElement {
    /// A node pattern: `(alias:Label {props})`
    Node(CypherNode),
    /// A relationship pattern: `-[r:TYPE*1..5]->`
    Rel(CypherRel),
}

/// A node pattern in a Cypher query.
///
/// Examples:
/// - `(n)` - anonymous node
/// - `(p:Person)` - labeled node
/// - `(p:Person:Employee {name: "Alice"})` - multiple labels with properties
#[derive(Clone, Debug, PartialEq)]
pub struct CypherNode {
    /// Variable name for binding.
    pub variable: Option<Ident>,
    /// Node labels (e.g., `:Person:Employee`).
    pub labels: Vec<Ident>,
    /// Inline property constraints.
    pub properties: Vec<Property>,
}

/// A relationship pattern in a Cypher query.
///
/// Examples:
/// - `-[r:KNOWS]->` - typed outgoing relationship
/// - `<-[:WORKS_AT]-` - typed incoming relationship
/// - `-[*1..5]-` - variable-length undirected
/// - `-[:KNOWS|FOLLOWS]->` - multiple types
#[derive(Clone, Debug, PartialEq)]
pub struct CypherRel {
    /// Variable name for binding.
    pub variable: Option<Ident>,
    /// Relationship types (OR semantics).
    pub rel_types: Vec<Ident>,
    /// Relationship direction.
    pub direction: CypherDirection,
    /// Variable-length specification.
    pub var_length: Option<CypherVarLength>,
    /// Inline property constraints.
    pub properties: Vec<Property>,
}

/// Relationship direction in a pattern.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CypherDirection {
    /// Outgoing: `-[]->`
    #[default]
    Outgoing,
    /// Incoming: `<-[]-`
    Incoming,
    /// Undirected: `-[]-`
    Undirected,
}

/// Variable-length relationship specification.
///
/// Examples:
/// - `*` - any length (0 or more)
/// - `*3` - exactly 3 hops
/// - `*1..5` - 1 to 5 hops
/// - `*..5` - 0 to 5 hops
/// - `*3..` - 3 or more hops
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CypherVarLength {
    /// Minimum hops (defaults to 1).
    pub min: Option<u32>,
    /// Maximum hops (None = unlimited).
    pub max: Option<u32>,
}

impl CypherVarLength {
    /// Create a fixed-length spec (exactly n hops).
    #[must_use]
    pub const fn exact(n: u32) -> Self {
        Self {
            min: Some(n),
            max: Some(n),
        }
    }

    /// Create a range spec.
    #[must_use]
    pub const fn range(min: Option<u32>, max: Option<u32>) -> Self {
        Self { min, max }
    }

    /// Create an unbounded spec (any length).
    #[must_use]
    pub const fn unbounded() -> Self {
        Self {
            min: None,
            max: None,
        }
    }
}

/// RETURN clause in a Cypher query.
#[derive(Clone, Debug, PartialEq)]
pub struct CypherReturn {
    /// Whether DISTINCT is specified.
    pub distinct: bool,
    /// Return items (expressions with optional aliases).
    pub items: Vec<CypherReturnItem>,
}

/// A single item in a RETURN clause.
///
/// Examples:
/// - `p.name` - property access
/// - `COUNT(p) AS total` - aggregation with alias
#[derive(Clone, Debug, PartialEq)]
pub struct CypherReturnItem {
    /// The expression to return.
    pub expr: Expr,
    /// Optional alias (AS name).
    pub alias: Option<Ident>,
}

/// CREATE statement for graph mutations.
///
/// Examples:
/// - `CREATE (p:Person {name: "Bob"})`
/// - `CREATE (a)-[:KNOWS]->(b)`
#[derive(Clone, Debug, PartialEq)]
pub struct CypherCreateStmt {
    /// Patterns to create.
    pub patterns: Vec<CypherPattern>,
}

/// DELETE statement for graph mutations.
///
/// Examples:
/// - `DELETE n` - delete node (fails if has relationships)
/// - `DETACH DELETE n` - delete node and all its relationships
#[derive(Clone, Debug, PartialEq)]
pub struct CypherDeleteStmt {
    /// Whether DETACH is specified (also delete relationships).
    pub detach: bool,
    /// Variables to delete.
    pub variables: Vec<Expr>,
}

/// MERGE statement for upsert operations.
///
/// Example:
/// - `MERGE (p:Person {name: "Alice"}) ON CREATE SET p.created = timestamp()`
#[derive(Clone, Debug, PartialEq)]
pub struct CypherMergeStmt {
    /// Pattern to match or create.
    pub pattern: CypherPattern,
    /// ON CREATE actions.
    pub on_create: Vec<CypherSetItem>,
    /// ON MATCH actions.
    pub on_match: Vec<CypherSetItem>,
}

/// A SET item in MERGE ON CREATE/ON MATCH.
#[derive(Clone, Debug, PartialEq)]
pub struct CypherSetItem {
    /// Property to set (e.g., `p.name`).
    pub property: Expr,
    /// Value to set.
    pub value: Expr,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cypher_var_length_exact() {
        let vl = CypherVarLength::exact(3);
        assert_eq!(vl.min, Some(3));
        assert_eq!(vl.max, Some(3));
    }

    #[test]
    fn test_cypher_var_length_range() {
        let vl = CypherVarLength::range(Some(1), Some(5));
        assert_eq!(vl.min, Some(1));
        assert_eq!(vl.max, Some(5));
    }

    #[test]
    fn test_cypher_var_length_unbounded() {
        let vl = CypherVarLength::unbounded();
        assert_eq!(vl.min, None);
        assert_eq!(vl.max, None);
    }

    #[test]
    fn test_cypher_direction_default() {
        let dir = CypherDirection::default();
        assert_eq!(dir, CypherDirection::Outgoing);
    }

    #[test]
    fn test_cypher_node_empty() {
        let node = CypherNode {
            variable: None,
            labels: Vec::new(),
            properties: Vec::new(),
        };
        assert!(node.variable.is_none());
        assert!(node.labels.is_empty());
    }

    #[test]
    fn test_cypher_pattern_empty() {
        let pattern = CypherPattern {
            variable: None,
            elements: Vec::new(),
        };
        assert!(pattern.elements.is_empty());
    }
}
