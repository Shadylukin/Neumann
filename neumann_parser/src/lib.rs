//! Neumann Parser - A hand-written recursive descent parser for the Neumann query language.
//!
//! Supports SQL statements, graph commands, vector operations, and unified queries.
//!
//! # Example
//!
//! ```
//! use neumann_parser::{parse, tokenize};
//!
//! // Parse a SQL statement
//! let stmt = parse("SELECT * FROM users WHERE id = 1").unwrap();
//!
//! // Tokenize source text
//! let tokens = tokenize("SELECT id, name FROM users");
//! ```

pub mod ast;
pub mod error;
pub mod expr;
pub mod lexer;
pub mod parser;
pub mod span;
pub mod token;

pub use ast::*;
pub use error::{Errors, ParseError, ParseErrorKind, ParseResult};
pub use expr::{parse_expr, ExprParser};
pub use lexer::{tokenize, Lexer};
pub use parser::{parse, parse_all, Parser};
pub use span::{get_line, line_col, line_number, BytePos, Span, Spanned};
pub use token::{Token, TokenKind};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_api_parse() {
        let stmt = parse("SELECT * FROM users").unwrap();
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_public_api_tokenize() {
        let tokens = tokenize("SELECT * FROM users");
        assert!(tokens.len() > 0);
        assert!(matches!(tokens[0].kind, TokenKind::Select));
    }

    #[test]
    fn test_public_api_parse_expr() {
        let expr = parse_expr("1 + 2 * 3").unwrap();
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Add, _)));
    }

    #[test]
    fn test_public_api_parse_all() {
        let stmts = parse_all("SELECT 1; SELECT 2").unwrap();
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_error_formatting() {
        let result = parse("SELCT * FROM users");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let formatted = err.format_with_source("SELCT * FROM users");
        assert!(formatted.contains("error"));
    }

    #[test]
    fn test_span_utilities() {
        let source = "SELECT\nFROM";
        assert_eq!(line_number(source, BytePos(7)), 2);
        assert_eq!(line_col(source, BytePos(7)), (2, 1));
        assert_eq!(get_line(source, BytePos(7)), "FROM");
    }

    #[test]
    fn test_graph_statement() {
        let stmt = parse("NODE CREATE person {name: 'Alice'}").unwrap();
        assert!(matches!(stmt.kind, StatementKind::Node(_)));
    }

    #[test]
    fn test_vector_statement() {
        let stmt = parse("SIMILAR 'query' LIMIT 10").unwrap();
        assert!(matches!(stmt.kind, StatementKind::Similar(_)));
    }

    #[test]
    fn test_unified_statement() {
        let stmt = parse("FIND NODE person WHERE age > 18").unwrap();
        assert!(matches!(stmt.kind, StatementKind::Find(_)));
    }

    #[test]
    fn test_entity_create_statement() {
        let stmt = parse("ENTITY CREATE 'user:1' { name: 'Alice' }").unwrap();
        assert!(matches!(stmt.kind, StatementKind::Entity(_)));
    }

    #[test]
    fn test_entity_create_with_embedding() {
        let stmt = parse("ENTITY CREATE 'doc:1' { title: 'Test' } EMBEDDING [1.0, 0.0]").unwrap();
        if let StatementKind::Entity(EntityStmt {
            operation: EntityOp::Create { embedding, .. },
        }) = stmt.kind
        {
            assert!(embedding.is_some());
            assert_eq!(embedding.unwrap().len(), 2);
        } else {
            panic!("expected ENTITY CREATE");
        }
    }

    #[test]
    fn test_entity_connect_statement() {
        let stmt = parse("ENTITY CONNECT 'from' -> 'to' : follows").unwrap();
        if let StatementKind::Entity(EntityStmt {
            operation: EntityOp::Connect { edge_type, .. },
        }) = stmt.kind
        {
            assert_eq!(edge_type.name, "follows");
        } else {
            panic!("expected ENTITY CONNECT");
        }
    }

    #[test]
    fn test_similar_connected_to() {
        let stmt = parse("SIMILAR 'key' CONNECTED TO 'hub' LIMIT 10").unwrap();
        if let StatementKind::Similar(similar) = stmt.kind {
            assert!(similar.connected_to.is_some());
            assert!(similar.limit.is_some());
        } else {
            panic!("expected SIMILAR");
        }
    }

    #[test]
    fn test_neighbors_by_similarity() {
        let stmt = parse("NEIGHBORS 'entity' BY SIMILAR [1.0, 0.0] LIMIT 5").unwrap();
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert!(neighbors.by_similarity.is_some());
            assert_eq!(neighbors.by_similarity.unwrap().len(), 2);
        } else {
            panic!("expected NEIGHBORS");
        }
    }

    #[test]
    fn test_lexer_api() {
        let mut lexer = Lexer::new("SELECT 1");
        let token = lexer.next_token();
        assert!(matches!(token.kind, TokenKind::Select));
        assert_eq!(lexer.source(), "SELECT 1");
    }

    #[test]
    fn test_parser_api() {
        let mut parser = Parser::new("SELECT 1; SELECT 2");
        let stmt1 = parser.parse_statement().unwrap();
        assert!(matches!(stmt1.kind, StatementKind::Select(_)));
        let stmt2 = parser.parse_statement().unwrap();
        assert!(matches!(stmt2.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expr_parser_api() {
        let mut parser = ExprParser::new("1 + 2");
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Add, _)));
    }

    #[test]
    fn test_complex_sql() {
        let sql = r#"
            SELECT u.name, COUNT(o.id) AS order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.active = TRUE
            GROUP BY u.name
            HAVING COUNT(o.id) > 0
            ORDER BY order_count DESC
            LIMIT 10
        "#;
        let stmt = parse(sql).unwrap();
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.distinct == false);
            assert!(select.from.is_some());
            assert!(select.where_clause.is_some());
            assert!(!select.group_by.is_empty());
            assert!(select.having.is_some());
            assert!(!select.order_by.is_empty());
            assert!(select.limit.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_insert_values() {
        let stmt =
            parse("INSERT INTO users (name, age) VALUES ('Bob', 25), ('Carol', 30)").unwrap();
        if let StatementKind::Insert(insert) = stmt.kind {
            assert_eq!(insert.table.name, "users");
            if let InsertSource::Values(rows) = insert.source {
                assert_eq!(rows.len(), 2);
            } else {
                panic!("expected VALUES");
            }
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_create_table_full() {
        let sql = r#"
            CREATE TABLE IF NOT EXISTS orders (
                id INT PRIMARY KEY,
                user_id INT NOT NULL REFERENCES users(id),
                total DECIMAL(10, 2) DEFAULT 0.00,
                created_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        "#;
        let stmt = parse(sql).unwrap();
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create.if_not_exists);
            assert_eq!(create.columns.len(), 4);
            assert_eq!(create.constraints.len(), 1);
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_edge_create_full() {
        let stmt = parse("EDGE CREATE 1 -> 2 : FOLLOWS {since: 2023, weight: 0.8}").unwrap();
        if let StatementKind::Edge(EdgeStmt {
            operation:
                EdgeOp::Create {
                    edge_type,
                    properties,
                    ..
                },
        }) = stmt.kind
        {
            assert_eq!(edge_type.name, "FOLLOWS");
            assert_eq!(properties.len(), 2);
        } else {
            panic!("expected EDGE CREATE");
        }
    }

    #[test]
    fn test_spanned_type() {
        let spanned = Spanned::new(42, Span::from_offsets(0, 2));
        assert_eq!(spanned.node, 42);
        assert_eq!(spanned.span.len(), 2);
    }

    #[test]
    fn test_errors_collection() {
        let mut errors = Errors::new();
        errors.push(ParseError::invalid("test", Span::from_offsets(0, 1)));
        assert!(!errors.is_empty());
        assert_eq!(errors.len(), 1);
    }

    // Chain statement tests
    #[test]
    fn test_chain_begin() {
        let stmt = parse("BEGIN CHAIN TRANSACTION").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::Begin));
        } else {
            panic!("expected CHAIN BEGIN");
        }
    }

    #[test]
    fn test_chain_commit() {
        let stmt = parse("COMMIT CHAIN").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::Commit));
        } else {
            panic!("expected CHAIN COMMIT");
        }
    }

    #[test]
    fn test_chain_rollback() {
        let stmt = parse("ROLLBACK CHAIN TO 100").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::Rollback { height } = chain.operation {
                if let ExprKind::Literal(Literal::Integer(h)) = height.kind {
                    assert_eq!(h, 100);
                } else {
                    panic!("expected integer");
                }
            } else {
                panic!("expected CHAIN ROLLBACK");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_chain_history() {
        let stmt = parse("CHAIN HISTORY 'users:123'").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::History { key } = chain.operation {
                if let ExprKind::Literal(Literal::String(k)) = key.kind {
                    assert_eq!(k, "users:123");
                } else {
                    panic!("expected string");
                }
            } else {
                panic!("expected CHAIN HISTORY");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_chain_similar() {
        let stmt = parse("CHAIN SIMILAR [1.0, 2.0, 3.0] LIMIT 10").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::Similar { embedding, limit } = chain.operation {
                assert_eq!(embedding.len(), 3);
                assert!(limit.is_some());
            } else {
                panic!("expected CHAIN SIMILAR");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_chain_drift() {
        let stmt = parse("CHAIN DRIFT FROM 0 TO 1000").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::Drift { from_height, to_height } = chain.operation {
                if let ExprKind::Literal(Literal::Integer(f)) = from_height.kind {
                    assert_eq!(f, 0);
                }
                if let ExprKind::Literal(Literal::Integer(t)) = to_height.kind {
                    assert_eq!(t, 1000);
                }
            } else {
                panic!("expected CHAIN DRIFT");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_chain_height() {
        let stmt = parse("CHAIN HEIGHT").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::Height));
        } else {
            panic!("expected CHAIN HEIGHT");
        }
    }

    #[test]
    fn test_chain_tip() {
        let stmt = parse("CHAIN TIP").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::Tip));
        } else {
            panic!("expected CHAIN TIP");
        }
    }

    #[test]
    fn test_chain_block() {
        let stmt = parse("CHAIN BLOCK 42").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::Block { height } = chain.operation {
                if let ExprKind::Literal(Literal::Integer(h)) = height.kind {
                    assert_eq!(h, 42);
                }
            } else {
                panic!("expected CHAIN BLOCK");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_chain_verify() {
        let stmt = parse("CHAIN VERIFY").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::Verify));
        } else {
            panic!("expected CHAIN VERIFY");
        }
    }

    #[test]
    fn test_show_codebook_global() {
        let stmt = parse("SHOW CODEBOOK GLOBAL").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::ShowCodebookGlobal));
        } else {
            panic!("expected SHOW CODEBOOK GLOBAL");
        }
    }

    #[test]
    fn test_show_codebook_local() {
        let stmt = parse("SHOW CODEBOOK LOCAL 'users'").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            if let ChainOp::ShowCodebookLocal { domain } = chain.operation {
                if let ExprKind::Literal(Literal::String(d)) = domain.kind {
                    assert_eq!(d, "users");
                }
            } else {
                panic!("expected SHOW CODEBOOK LOCAL");
            }
        } else {
            panic!("expected CHAIN statement");
        }
    }

    #[test]
    fn test_analyze_codebook_transitions() {
        let stmt = parse("ANALYZE CODEBOOK TRANSITIONS").unwrap();
        if let StatementKind::Chain(chain) = stmt.kind {
            assert!(matches!(chain.operation, ChainOp::AnalyzeTransitions));
        } else {
            panic!("expected ANALYZE CODEBOOK TRANSITIONS");
        }
    }
}
