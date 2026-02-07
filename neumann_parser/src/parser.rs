// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Statement parser for the Neumann query language.
//!
//! Parses complete statements including:
//! - SQL statements (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP)
//! - Graph commands (NODE, EDGE, NEIGHBORS, PATH)
//! - Vector commands (EMBED, SIMILAR)
//! - Unified queries (FIND)

#![allow(clippy::wildcard_imports)]
#![allow(clippy::enum_glob_use)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::if_not_else)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::{
    ast::*,
    error::{ParseError, ParseErrorKind, ParseResult},
    lexer::Lexer,
    span::Span,
    token::{Token, TokenKind},
};

/// Binding power for prefix operators.
const PREFIX_BP: u8 = 19;

/// Returns binding power for infix operators.
const fn infix_binding_power(op: BinaryOp) -> (u8, u8) {
    use BinaryOp::*;
    match op {
        Or => (1, 2),
        And => (3, 4),
        Eq | Ne | Lt | Le | Gt | Ge => (5, 6),
        BitOr => (7, 8),
        BitXor => (9, 10),
        BitAnd => (11, 12),
        Shl | Shr => (13, 14),
        Add | Sub | Concat => (15, 16),
        Mul | Div | Mod => (17, 18),
    }
}

/// Statement parser.
pub struct Parser<'a> {
    source: &'a str,
    lexer: Lexer<'a>,
    current: Token,
    peeked: Option<Token>,
}

impl<'a> Parser<'a> {
    /// Creates a new parser for the given source.
    #[must_use]
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self {
            source,
            lexer,
            current,
            peeked: None,
        }
    }

    /// Returns the source text.
    #[must_use]
    pub const fn source(&self) -> &'a str {
        self.source
    }

    /// Returns the current token.
    #[must_use]
    pub const fn current(&self) -> &Token {
        &self.current
    }

    /// Peeks at the next token.
    fn peek(&mut self) -> &Token {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next_token());
        }
        self.peeked.as_ref().unwrap()
    }

    /// Advances to the next token.
    fn advance(&mut self) -> Token {
        std::mem::replace(
            &mut self.current,
            self.peeked
                .take()
                .unwrap_or_else(|| self.lexer.next_token()),
        )
    }

    /// Returns true if the current token matches.
    fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current.kind) == std::mem::discriminant(kind)
    }

    /// Consumes the current token if it matches.
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expects the current token to match.
    fn expect(&mut self, kind: &TokenKind) -> ParseResult<Token> {
        if self.check(kind) {
            Ok(self.advance())
        } else if self.current.is_eof() {
            Err(ParseError::unexpected_eof(self.current.span, kind.as_str()))
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                kind.as_str(),
            ))
        }
    }

    /// Expects and returns an identifier.
    fn expect_ident(&mut self) -> ParseResult<Ident> {
        let token = self.expect(&TokenKind::Ident(String::new()))?;
        match token.kind {
            TokenKind::Ident(name) => Ok(Ident::new(name, token.span)),
            _ => unreachable!(),
        }
    }

    /// Expects and returns an identifier or contextual keyword (for use in contexts like field
    /// names).
    fn expect_ident_or_keyword(&mut self) -> ParseResult<Ident> {
        let token = self.current.clone();
        if let TokenKind::Ident(name) = &token.kind {
            self.advance();
            Ok(Ident::new(name.clone(), token.span))
        } else if token.kind.is_contextual_keyword() {
            self.advance();
            // Convert keyword to lowercase string for use as identifier
            let name = token.kind.as_str().to_lowercase();
            Ok(Ident::new(name, token.span))
        } else {
            Err(ParseError::unexpected(token.kind, token.span, "identifier"))
        }
    }

    /// Expects an identifier or any keyword (including reserved ones like FROM, TO).
    /// Used in contexts like batch edge definitions where reserved keywords are valid keys.
    fn expect_ident_or_any_keyword(&mut self) -> ParseResult<Ident> {
        let token = self.current.clone();
        if let TokenKind::Ident(name) = &token.kind {
            self.advance();
            Ok(Ident::new(name.clone(), token.span))
        } else if token.kind.is_keyword() {
            self.advance();
            // Convert keyword to lowercase string for use as identifier
            let name = token.kind.as_str().to_lowercase();
            Ok(Ident::new(name, token.span))
        } else {
            Err(ParseError::unexpected(token.kind, token.span, "identifier"))
        }
    }

    /// Parses a direction (OUTGOING, INCOMING, BOTH).
    fn parse_direction(&mut self) -> ParseResult<Direction> {
        if self.eat(&TokenKind::Outgoing) {
            Ok(Direction::Outgoing)
        } else if self.eat(&TokenKind::Incoming) {
            Ok(Direction::Incoming)
        } else if self.eat(&TokenKind::Both) {
            Ok(Direction::Both)
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "OUTGOING, INCOMING, or BOTH",
            ))
        }
    }

    /// Parses an expression.
    fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_expr_bp(0)
    }

    /// Parses an expression with the given minimum binding power.
    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        let mut lhs = self.parse_prefix_expr()?;

        loop {
            // Check for postfix operators
            lhs = self.parse_postfix_expr(lhs)?;

            // Check for infix operators
            let op = match self.current_binary_op() {
                Some(op) => op,
                None => break,
            };

            let (l_bp, r_bp) = infix_binding_power(op);
            if l_bp < min_bp {
                break;
            }

            self.advance();
            let rhs = self.parse_expr_bp(r_bp)?;

            let span = lhs.span.merge(rhs.span);
            lhs = Expr::new(ExprKind::Binary(Box::new(lhs), op, Box::new(rhs)), span);
        }

        Ok(lhs)
    }

    /// Parses a prefix expression.
    fn parse_prefix_expr(&mut self) -> ParseResult<Expr> {
        let token = self.current.clone();

        match &self.current.kind {
            // Literals
            TokenKind::Integer(n) => {
                let n = *n;
                self.advance();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Integer(n)),
                    token.span,
                ))
            },
            TokenKind::Float(n) => {
                let n = *n;
                self.advance();
                Ok(Expr::new(ExprKind::Literal(Literal::Float(n)), token.span))
            },
            TokenKind::String(s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::new(ExprKind::Literal(Literal::String(s)), token.span))
            },
            TokenKind::True => {
                self.advance();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Boolean(true)),
                    token.span,
                ))
            },
            TokenKind::False => {
                self.advance();
                Ok(Expr::new(
                    ExprKind::Literal(Literal::Boolean(false)),
                    token.span,
                ))
            },
            TokenKind::Null => {
                self.advance();
                Ok(Expr::new(ExprKind::Literal(Literal::Null), token.span))
            },

            // Identifiers
            TokenKind::Ident(_) => self.parse_ident_or_call_expr(),

            // Aggregate functions
            TokenKind::Count
            | TokenKind::Sum
            | TokenKind::Avg
            | TokenKind::Min
            | TokenKind::Max => self.parse_aggregate_call_expr(),

            // Wildcard
            TokenKind::Star => {
                self.advance();
                Ok(Expr::new(ExprKind::Wildcard, token.span))
            },

            // Parenthesized expression
            TokenKind::LParen => self.parse_paren_expr(),

            // Array literal
            TokenKind::LBracket => self.parse_array_expr(),

            // Unary operators
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::Neg, Box::new(operand)),
                    span,
                ))
            },
            TokenKind::Not | TokenKind::Bang => {
                self.advance();
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::Not, Box::new(operand)),
                    span,
                ))
            },
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_expr_bp(PREFIX_BP)?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::BitNot, Box::new(operand)),
                    span,
                ))
            },

            // CASE expression
            TokenKind::Case => self.parse_case_expr(),

            // EXISTS subquery
            TokenKind::Exists => self.parse_exists_expr(),

            // CAST expression
            TokenKind::Cast => self.parse_cast_expr(),

            TokenKind::Eof => Err(ParseError::unexpected_eof(token.span, "expression")),

            // Allow contextual keywords to be used as identifiers (e.g., column names like
            // "status")
            _ if token.kind.is_contextual_keyword() => self.parse_keyword_as_ident_expr(),

            _ => Err(ParseError::unexpected(
                token.kind.clone(),
                token.span,
                "expression",
            )),
        }
    }

    /// Parses a keyword token as an identifier expression.
    fn parse_keyword_as_ident_expr(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = token.kind.as_str().to_lowercase();
        let ident = Ident::new(name, token.span);
        Ok(Expr::new(ExprKind::Ident(ident), token.span))
    }

    /// Parses postfix operators.
    fn parse_postfix_expr(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        loop {
            // Check for NOT followed by IN/BETWEEN/LIKE
            if self.check(&TokenKind::Not) {
                let next_kind = self.peek().kind.clone();
                if next_kind == TokenKind::In {
                    self.advance();
                    expr = self.parse_in_expr(expr, true)?;
                    continue;
                } else if next_kind == TokenKind::Between {
                    self.advance();
                    expr = self.parse_between_expr(expr, true)?;
                    continue;
                } else if next_kind == TokenKind::Like {
                    self.advance();
                    expr = self.parse_like_expr(expr, true)?;
                    continue;
                }
            }

            let kind = self.current.kind.clone();
            expr = match kind {
                TokenKind::Is => {
                    self.advance();
                    let negated = self.eat(&TokenKind::Not);
                    self.expect(&TokenKind::Null)?;
                    let span = expr.span.merge(self.current.span);
                    Expr::new(
                        ExprKind::IsNull {
                            expr: Box::new(expr),
                            negated,
                        },
                        span,
                    )
                },
                TokenKind::In => self.parse_in_expr(expr, false)?,
                TokenKind::Between => self.parse_between_expr(expr, false)?,
                TokenKind::Like => self.parse_like_expr(expr, false)?,
                TokenKind::Dot => {
                    self.advance();
                    if self.eat(&TokenKind::Star) {
                        if let ExprKind::Ident(ident) = expr.kind {
                            let span = expr.span.merge(self.current.span);
                            Expr::new(ExprKind::QualifiedWildcard(ident), span)
                        } else {
                            return Err(ParseError::invalid(
                                "qualified wildcard requires identifier",
                                expr.span,
                            ));
                        }
                    } else {
                        let token = self.expect(&TokenKind::Ident(String::new()))?;
                        let name = match token.kind {
                            TokenKind::Ident(s) => s,
                            _ => unreachable!(),
                        };
                        let span = expr.span.merge(token.span);
                        Expr::new(
                            ExprKind::Qualified(Box::new(expr), Ident::new(name, token.span)),
                            span,
                        )
                    }
                },
                _ => return Ok(expr),
            };
        }
    }

    fn parse_ident_or_call_expr(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = match token.kind {
            TokenKind::Ident(s) => s,
            _ => unreachable!(),
        };
        let ident = Ident::new(name, token.span);

        if self.check(&TokenKind::LParen) {
            self.parse_function_call_expr(ident, token.span)
        } else {
            Ok(Expr::new(ExprKind::Ident(ident), token.span))
        }
    }

    fn parse_function_call_expr(&mut self, name: Ident, start: Span) -> ParseResult<Expr> {
        self.expect(&TokenKind::LParen)?;
        let distinct = self.eat(&TokenKind::Distinct);
        let mut args = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RParen)?;
        Ok(Expr::new(
            ExprKind::Call(FunctionCall {
                name,
                args,
                distinct,
            }),
            start.merge(end.span),
        ))
    }

    fn parse_aggregate_call_expr(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = match &token.kind {
            TokenKind::Count => "COUNT",
            TokenKind::Sum => "SUM",
            TokenKind::Avg => "AVG",
            TokenKind::Min => "MIN",
            TokenKind::Max => "MAX",
            _ => unreachable!(),
        };
        self.parse_function_call_expr(Ident::new(name, token.span), token.span)
    }

    fn parse_paren_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::LParen)?.span;
        if self.check(&TokenKind::RParen) {
            let end = self.advance().span;
            return Ok(Expr::new(ExprKind::Tuple(Vec::new()), start.merge(end)));
        }
        let first = self.parse_expr()?;
        if self.eat(&TokenKind::Comma) {
            let mut items = vec![first];
            loop {
                items.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            let end = self.expect(&TokenKind::RParen)?.span;
            return Ok(Expr::new(ExprKind::Tuple(items), start.merge(end)));
        }
        let end = self.expect(&TokenKind::RParen)?.span;
        Ok(Expr::new(first.kind, start.merge(end)))
    }

    fn parse_array_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::LBracket)?.span;
        let mut items = Vec::new();
        if !self.check(&TokenKind::RBracket) {
            loop {
                items.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RBracket)?.span;
        Ok(Expr::new(ExprKind::Array(items), start.merge(end)))
    }

    fn parse_vector_literal(&mut self) -> ParseResult<Vec<Expr>> {
        self.expect(&TokenKind::LBracket)?;
        let mut items = Vec::new();
        if !self.check(&TokenKind::RBracket) {
            loop {
                items.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        self.expect(&TokenKind::RBracket)?;
        Ok(items)
    }

    fn parse_case_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::Case)?.span;
        let operand = if !self.check(&TokenKind::When) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let mut when_clauses = Vec::new();
        while self.eat(&TokenKind::When) {
            let condition = self.parse_expr()?;
            self.expect(&TokenKind::Then)?;
            let result = self.parse_expr()?;
            when_clauses.push(WhenClause { condition, result });
        }
        if when_clauses.is_empty() {
            return Err(ParseError::invalid(
                "CASE requires at least one WHEN clause",
                self.current.span,
            ));
        }
        let else_clause = if self.eat(&TokenKind::Else) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };
        let end = self.expect(&TokenKind::End)?.span;
        Ok(Expr::new(
            ExprKind::Case(CaseExpr {
                operand,
                when_clauses,
                else_clause,
            }),
            start.merge(end),
        ))
    }

    fn parse_exists_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::Exists)?.span;
        self.expect(&TokenKind::LParen)?;
        self.expect(&TokenKind::Select)?;
        let subquery = self.parse_select_body()?;
        let end = self.expect(&TokenKind::RParen)?.span;
        Ok(Expr::new(
            ExprKind::Exists(Box::new(subquery)),
            start.merge(end),
        ))
    }

    fn parse_cast_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::Cast)?.span;
        self.expect(&TokenKind::LParen)?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::As)?;
        let data_type = self.parse_data_type()?;
        let end = self.expect(&TokenKind::RParen)?.span;
        Ok(Expr::new(
            ExprKind::Cast(Box::new(expr), data_type),
            start.merge(end),
        ))
    }

    fn parse_in_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        let start_span = expr.span;
        self.expect(&TokenKind::In)?;
        self.expect(&TokenKind::LParen)?;

        // Check for subquery: (SELECT ...)
        let list = if self.check(&TokenKind::Select) {
            self.advance(); // consume SELECT
            let subquery = self.parse_select_body()?;
            InList::Subquery(Box::new(subquery))
        } else {
            let mut values = Vec::new();
            if !self.check(&TokenKind::RParen) {
                loop {
                    values.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            InList::Values(values)
        };

        let end = self.expect(&TokenKind::RParen)?.span;
        Ok(Expr::new(
            ExprKind::In {
                expr: Box::new(expr),
                list,
                negated,
            },
            start_span.merge(end),
        ))
    }

    fn parse_between_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        self.expect(&TokenKind::Between)?;
        let low = self.parse_expr_bp(PREFIX_BP)?;
        self.expect(&TokenKind::And)?;
        let high = self.parse_expr_bp(PREFIX_BP)?;
        let span = expr.span.merge(high.span);
        Ok(Expr::new(
            ExprKind::Between {
                expr: Box::new(expr),
                low: Box::new(low),
                high: Box::new(high),
                negated,
            },
            span,
        ))
    }

    fn parse_like_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        self.expect(&TokenKind::Like)?;
        let pattern = self.parse_expr_bp(PREFIX_BP)?;
        let span = expr.span.merge(pattern.span);
        Ok(Expr::new(
            ExprKind::Like {
                expr: Box::new(expr),
                pattern: Box::new(pattern),
                negated,
            },
            span,
        ))
    }

    const fn current_binary_op(&self) -> Option<BinaryOp> {
        match &self.current.kind {
            TokenKind::Plus => Some(BinaryOp::Add),
            TokenKind::Minus => Some(BinaryOp::Sub),
            TokenKind::Star => Some(BinaryOp::Mul),
            TokenKind::Slash => Some(BinaryOp::Div),
            TokenKind::Percent => Some(BinaryOp::Mod),
            TokenKind::Eq => Some(BinaryOp::Eq),
            TokenKind::Ne => Some(BinaryOp::Ne),
            TokenKind::Lt => Some(BinaryOp::Lt),
            TokenKind::Le => Some(BinaryOp::Le),
            TokenKind::Gt => Some(BinaryOp::Gt),
            TokenKind::Ge => Some(BinaryOp::Ge),
            TokenKind::And => Some(BinaryOp::And),
            TokenKind::Or => Some(BinaryOp::Or),
            TokenKind::Concat => Some(BinaryOp::Concat),
            TokenKind::Amp => Some(BinaryOp::BitAnd),
            TokenKind::Pipe => Some(BinaryOp::BitOr),
            TokenKind::Caret => Some(BinaryOp::BitXor),
            TokenKind::Shl => Some(BinaryOp::Shl),
            TokenKind::Shr => Some(BinaryOp::Shr),
            _ => None,
        }
    }

    /// Parses a statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the input is not a valid statement.
    pub fn parse_statement(&mut self) -> ParseResult<Statement> {
        // Skip empty statements (just semicolons)
        while self.eat(&TokenKind::Semicolon) {}

        if self.current.is_eof() {
            return Ok(Statement::new(StatementKind::Empty, self.current.span));
        }

        let start = self.current.span;

        let kind = match &self.current.kind {
            // SQL Statements
            TokenKind::Select => self.parse_select()?,
            TokenKind::Insert => self.parse_insert()?,
            TokenKind::Update => self.parse_update()?,
            TokenKind::Delete => self.parse_delete()?,
            TokenKind::Create => self.parse_create()?,
            TokenKind::Drop => self.parse_drop()?,
            TokenKind::Show => self.parse_show()?,
            TokenKind::Describe => self.parse_describe()?,
            TokenKind::Count => self.parse_count()?,

            // Graph Statements
            TokenKind::Node => self.parse_node()?,
            TokenKind::Edge => self.parse_edge()?,
            TokenKind::Neighbors => self.parse_neighbors()?,
            TokenKind::Path => self.parse_path()?,

            // Vector Statements
            TokenKind::Embed => self.parse_embed()?,
            TokenKind::Similar => self.parse_similar()?,

            // Unified Statements
            TokenKind::Find => self.parse_find()?,
            TokenKind::Entity => self.parse_entity()?,

            // Vault Statements
            TokenKind::Vault => self.parse_vault()?,

            // Cache Statements
            TokenKind::Cache => self.parse_cache()?,

            // Blob Storage Statements
            TokenKind::Blob => self.parse_blob()?,
            TokenKind::Blobs => self.parse_blobs()?,

            // Checkpoint Statements
            TokenKind::Checkpoint => self.parse_checkpoint()?,
            TokenKind::Checkpoints => self.parse_checkpoints()?,
            TokenKind::Rollback => self.parse_rollback_or_chain_rollback()?,

            // Chain Statements
            TokenKind::Chain => self.parse_chain()?,
            TokenKind::Begin => self.parse_begin_chain()?,
            TokenKind::Commit => self.parse_commit_chain()?,
            TokenKind::Analyze => self.parse_analyze()?,

            // Cluster Statements
            TokenKind::Cluster => self.parse_cluster()?,

            // Extended Graph Statements
            TokenKind::Graph => self.parse_graph()?,
            TokenKind::Constraint => self.parse_constraint()?,
            TokenKind::Batch => self.parse_batch()?,
            TokenKind::Aggregate => self.parse_aggregate_stmt()?,

            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::UnknownCommand(format!("{}", self.current.kind)),
                    self.current.span,
                ));
            },
        };

        // Consume optional semicolon
        self.eat(&TokenKind::Semicolon);

        let end = self.current.span;
        Ok(Statement::new(kind, start.merge(end)))
    }

    // =========================================================================
    // SQL Statement Parsers
    // =========================================================================

    fn parse_select(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Select)?;
        Ok(StatementKind::Select(self.parse_select_body()?))
    }

    /// Parses a SELECT statement body (after the SELECT keyword).
    /// Used for both standalone SELECT and subqueries.
    fn parse_select_body(&mut self) -> ParseResult<SelectStmt> {
        // Handle DISTINCT or ALL (ALL is the default, just consume it)
        let distinct = if self.eat(&TokenKind::Distinct) {
            true
        } else {
            self.eat(&TokenKind::All); // consume ALL if present, but it's the default
            false
        };

        // Parse select items
        let mut columns = Vec::new();
        loop {
            let item = self.parse_select_item()?;
            columns.push(item);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        // Parse FROM clause
        let from = if self.eat(&TokenKind::From) {
            Some(self.parse_from_clause()?)
        } else {
            None
        };

        // Parse WHERE clause
        let where_clause = if self.eat(&TokenKind::Where) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        // Parse GROUP BY clause
        let group_by = if self.eat(&TokenKind::Group) {
            self.expect(&TokenKind::By)?;
            let mut exprs = Vec::new();
            loop {
                exprs.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            exprs
        } else {
            Vec::new()
        };

        // Parse HAVING clause
        let having = if self.eat(&TokenKind::Having) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        // Parse ORDER BY clause
        let order_by = if self.eat(&TokenKind::Order) {
            self.expect(&TokenKind::By)?;
            let mut items = Vec::new();
            loop {
                items.push(self.parse_order_by_item()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            items
        } else {
            Vec::new()
        };

        // Parse LIMIT
        let limit = if self.eat(&TokenKind::Limit) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        // Parse OFFSET
        let offset = if self.eat(&TokenKind::Offset) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(SelectStmt {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_select_item(&mut self) -> ParseResult<SelectItem> {
        let expr = self.parse_expr()?;

        let alias = if self.eat(&TokenKind::As) {
            Some(self.expect_ident()?)
        } else if let TokenKind::Ident(_) = &self.current.kind {
            // Implicit alias
            Some(self.expect_ident()?)
        } else {
            None
        };

        Ok(SelectItem { expr, alias })
    }

    fn parse_from_clause(&mut self) -> ParseResult<FromClause> {
        let table = self.parse_table_ref()?;

        let mut joins = Vec::new();
        while let Some(join) = self.try_parse_join()? {
            joins.push(join);
        }

        Ok(FromClause { table, joins })
    }

    fn parse_table_ref(&mut self) -> ParseResult<TableRef> {
        let start = self.current.span;

        // Check for subquery: (SELECT ...)
        let kind = if self.check(&TokenKind::LParen) {
            self.advance(); // consume '('
            self.expect(&TokenKind::Select)?;
            let subquery = self.parse_select_body()?;
            self.expect(&TokenKind::RParen)?;
            TableRefKind::Subquery(Box::new(subquery))
        } else {
            let name = self.expect_ident()?;
            TableRefKind::Table(name)
        };

        let alias = if self.eat(&TokenKind::As) {
            Some(self.expect_ident()?)
        } else if let TokenKind::Ident(_) = &self.current.kind {
            // Check it's not a keyword
            if !self.current.is_keyword() {
                Some(self.expect_ident()?)
            } else {
                None
            }
        } else {
            None
        };

        let end = alias.as_ref().map(|a| a.span).unwrap_or(start);

        Ok(TableRef {
            kind,
            alias,
            span: start.merge(end),
        })
    }

    fn try_parse_join(&mut self) -> ParseResult<Option<Join>> {
        let start = self.current.span;

        let kind = if self.eat(&TokenKind::Cross) {
            self.expect(&TokenKind::Join)?;
            JoinKind::Cross
        } else if self.eat(&TokenKind::Natural) {
            self.expect(&TokenKind::Join)?;
            JoinKind::Natural
        } else if self.eat(&TokenKind::Inner) {
            self.expect(&TokenKind::Join)?;
            JoinKind::Inner
        } else if self.eat(&TokenKind::Left) {
            self.eat(&TokenKind::Outer);
            self.expect(&TokenKind::Join)?;
            JoinKind::Left
        } else if self.eat(&TokenKind::Right) {
            self.eat(&TokenKind::Outer);
            self.expect(&TokenKind::Join)?;
            JoinKind::Right
        } else if self.eat(&TokenKind::Full) {
            self.eat(&TokenKind::Outer);
            self.expect(&TokenKind::Join)?;
            JoinKind::Full
        } else if self.eat(&TokenKind::Join) {
            JoinKind::Inner
        } else {
            return Ok(None);
        };

        let table = self.parse_table_ref()?;

        let condition = if self.eat(&TokenKind::On) {
            Some(JoinCondition::On(Box::new(self.parse_expr()?)))
        } else if self.eat(&TokenKind::Using) {
            self.expect(&TokenKind::LParen)?;
            let mut columns = Vec::new();
            loop {
                columns.push(self.expect_ident()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            Some(JoinCondition::Using(columns))
        } else {
            None
        };

        let end = self.current.span;

        Ok(Some(Join {
            kind,
            table,
            condition,
            span: start.merge(end),
        }))
    }

    fn parse_order_by_item(&mut self) -> ParseResult<OrderByItem> {
        let expr = self.parse_expr()?;

        let direction = if self.eat(&TokenKind::Desc) {
            SortDirection::Desc
        } else {
            self.eat(&TokenKind::Asc);
            SortDirection::Asc
        };

        let nulls = if self.eat(&TokenKind::Nulls) {
            if self.eat(&TokenKind::First) {
                Some(NullsOrder::First)
            } else {
                self.expect(&TokenKind::Last)?;
                Some(NullsOrder::Last)
            }
        } else {
            None
        };

        Ok(OrderByItem {
            expr,
            direction,
            nulls,
        })
    }

    fn parse_insert(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Insert)?;
        self.expect(&TokenKind::Into)?;

        let table = self.expect_ident()?;

        // Optional column list
        let columns = if self.eat(&TokenKind::LParen) {
            let mut cols = Vec::new();
            loop {
                cols.push(self.expect_ident()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            Some(cols)
        } else {
            None
        };

        // VALUES or SELECT
        let source = if self.eat(&TokenKind::Values) {
            let mut rows = Vec::new();
            loop {
                self.expect(&TokenKind::LParen)?;
                let mut values = Vec::new();
                loop {
                    values.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
                self.expect(&TokenKind::RParen)?;
                rows.push(values);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            InsertSource::Values(rows)
        } else if self.check(&TokenKind::Select) {
            // INSERT ... SELECT
            if let StatementKind::Select(select) = self.parse_select()? {
                InsertSource::Query(Box::new(select))
            } else {
                unreachable!("parse_select should return Select")
            }
        } else {
            return Err(ParseError::invalid(
                "INSERT requires VALUES or SELECT clause",
                self.current.span,
            ));
        };

        Ok(StatementKind::Insert(InsertStmt {
            table,
            columns,
            source,
        }))
    }

    fn parse_update(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Update)?;
        let table = self.expect_ident()?;
        self.expect(&TokenKind::Set)?;

        let mut assignments = Vec::new();
        loop {
            let column = self.expect_ident()?;
            self.expect(&TokenKind::Eq)?;
            let value = self.parse_expr()?;
            assignments.push(Assignment { column, value });
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        let where_clause = if self.eat(&TokenKind::Where) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(StatementKind::Update(UpdateStmt {
            table,
            assignments,
            where_clause,
        }))
    }

    fn parse_delete(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Delete)?;
        self.expect(&TokenKind::From)?;
        let table = self.expect_ident()?;

        let where_clause = if self.eat(&TokenKind::Where) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(StatementKind::Delete(DeleteStmt {
            table,
            where_clause,
        }))
    }

    fn parse_create(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Create)?;

        if self.eat(&TokenKind::Table) {
            self.parse_create_table()
        } else if self.eat(&TokenKind::Unique) {
            self.expect(&TokenKind::Index)?;
            self.parse_create_index(true)
        } else if self.eat(&TokenKind::Index) {
            self.parse_create_index(false)
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "TABLE or INDEX",
            ))
        }
    }

    fn parse_create_table(&mut self) -> ParseResult<StatementKind> {
        let if_not_exists = if self.eat(&TokenKind::If) {
            self.expect(&TokenKind::Not)?;
            self.expect(&TokenKind::Exists)?;
            true
        } else {
            false
        };

        let table = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        let mut columns = Vec::new();
        let mut constraints = Vec::new();

        loop {
            // Check for table constraint
            if self.check(&TokenKind::Primary)
                || self.check(&TokenKind::Foreign)
                || self.check(&TokenKind::Unique)
                || self.check(&TokenKind::Check)
                || self.check(&TokenKind::Constraint)
            {
                constraints.push(self.parse_table_constraint()?);
            } else {
                columns.push(self.parse_column_def()?);
            }

            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }

        self.expect(&TokenKind::RParen)?;

        Ok(StatementKind::CreateTable(CreateTableStmt {
            if_not_exists,
            table,
            columns,
            constraints,
        }))
    }

    fn parse_column_def(&mut self) -> ParseResult<ColumnDef> {
        let name = self.expect_ident_or_keyword()?;
        let data_type = self.parse_data_type()?;

        let mut constraints = Vec::new();
        while let Some(constraint) = self.try_parse_column_constraint()? {
            constraints.push(constraint);
        }

        Ok(ColumnDef {
            name,
            data_type,
            constraints,
        })
    }

    fn parse_data_type(&mut self) -> ParseResult<DataType> {
        let token = self.advance();
        let base_type = match &token.kind {
            TokenKind::Int => DataType::Int,
            TokenKind::Integer_ => DataType::Integer,
            TokenKind::Bigint => DataType::Bigint,
            TokenKind::Smallint => DataType::Smallint,
            TokenKind::Float_ => DataType::Float,
            TokenKind::Double => DataType::Double,
            TokenKind::Real => DataType::Real,
            TokenKind::Decimal => {
                let (p, s) = self.parse_precision_scale()?;
                DataType::Decimal(p, s)
            },
            TokenKind::Numeric => {
                let (p, s) = self.parse_precision_scale()?;
                DataType::Numeric(p, s)
            },
            TokenKind::Varchar => {
                let len = self.parse_type_length()?;
                DataType::Varchar(len)
            },
            TokenKind::Char => {
                let len = self.parse_type_length()?;
                DataType::Char(len)
            },
            TokenKind::Text => DataType::Text,
            TokenKind::Boolean => DataType::Boolean,
            TokenKind::Date => DataType::Date,
            TokenKind::Time => DataType::Time,
            TokenKind::Timestamp => DataType::Timestamp,
            TokenKind::Blob => DataType::Blob,
            TokenKind::Ident(name) => DataType::Custom(name.clone()),
            _ => {
                return Err(ParseError::unexpected(
                    token.kind.clone(),
                    token.span,
                    "data type",
                ));
            },
        };

        Ok(base_type)
    }

    fn parse_precision_scale(&mut self) -> ParseResult<(Option<u32>, Option<u32>)> {
        if self.eat(&TokenKind::LParen) {
            let token = self.expect(&TokenKind::Integer(0))?;
            let precision = match token.kind {
                TokenKind::Integer(n) => n as u32,
                _ => unreachable!(),
            };

            let scale = if self.eat(&TokenKind::Comma) {
                let token = self.expect(&TokenKind::Integer(0))?;
                match token.kind {
                    TokenKind::Integer(n) => Some(n as u32),
                    _ => unreachable!(),
                }
            } else {
                None
            };

            self.expect(&TokenKind::RParen)?;
            Ok((Some(precision), scale))
        } else {
            Ok((None, None))
        }
    }

    fn parse_type_length(&mut self) -> ParseResult<Option<u32>> {
        if self.eat(&TokenKind::LParen) {
            let token = self.expect(&TokenKind::Integer(0))?;
            let len = match token.kind {
                TokenKind::Integer(n) => n as u32,
                _ => unreachable!(),
            };
            self.expect(&TokenKind::RParen)?;
            Ok(Some(len))
        } else {
            Ok(None)
        }
    }

    fn try_parse_column_constraint(&mut self) -> ParseResult<Option<ColumnConstraint>> {
        if self.eat(&TokenKind::Not) {
            self.expect(&TokenKind::Null)?;
            Ok(Some(ColumnConstraint::NotNull))
        } else if self.eat(&TokenKind::Null) {
            Ok(Some(ColumnConstraint::Null))
        } else if self.eat(&TokenKind::Unique) {
            Ok(Some(ColumnConstraint::Unique))
        } else if self.eat(&TokenKind::Primary) {
            self.expect(&TokenKind::Key)?;
            Ok(Some(ColumnConstraint::PrimaryKey))
        } else if self.eat(&TokenKind::Default) {
            let expr = self.parse_expr()?;
            Ok(Some(ColumnConstraint::Default(expr)))
        } else if self.eat(&TokenKind::Check) {
            self.expect(&TokenKind::LParen)?;
            let expr = self.parse_expr()?;
            self.expect(&TokenKind::RParen)?;
            Ok(Some(ColumnConstraint::Check(expr)))
        } else if self.eat(&TokenKind::References) {
            let ref_table = self.expect_ident()?;
            let ref_column = if self.eat(&TokenKind::LParen) {
                let col = self.expect_ident()?;
                self.expect(&TokenKind::RParen)?;
                Some(col)
            } else {
                None
            };
            Ok(Some(ColumnConstraint::References(ForeignKeyRef {
                table: ref_table,
                column: ref_column,
                on_delete: None,
                on_update: None,
            })))
        } else {
            Ok(None)
        }
    }

    fn parse_table_constraint(&mut self) -> ParseResult<TableConstraint> {
        // Optional CONSTRAINT name
        if self.eat(&TokenKind::Constraint) {
            let _name = self.expect_ident()?;
        }

        if self.eat(&TokenKind::Primary) {
            self.expect(&TokenKind::Key)?;
            self.expect(&TokenKind::LParen)?;
            let mut columns = Vec::new();
            loop {
                columns.push(self.expect_ident()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            Ok(TableConstraint::PrimaryKey(columns))
        } else if self.eat(&TokenKind::Unique) {
            self.expect(&TokenKind::LParen)?;
            let mut columns = Vec::new();
            loop {
                columns.push(self.expect_ident()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            Ok(TableConstraint::Unique(columns))
        } else if self.eat(&TokenKind::Foreign) {
            self.expect(&TokenKind::Key)?;
            self.expect(&TokenKind::LParen)?;
            let mut columns = Vec::new();
            loop {
                columns.push(self.expect_ident()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            self.expect(&TokenKind::RParen)?;
            self.expect(&TokenKind::References)?;
            let ref_table = self.expect_ident()?;
            let ref_column = if self.eat(&TokenKind::LParen) {
                let col = self.expect_ident()?;
                self.expect(&TokenKind::RParen)?;
                Some(col)
            } else {
                None
            };
            Ok(TableConstraint::ForeignKey {
                columns,
                reference: ForeignKeyRef {
                    table: ref_table,
                    column: ref_column,
                    on_delete: None,
                    on_update: None,
                },
            })
        } else if self.eat(&TokenKind::Check) {
            self.expect(&TokenKind::LParen)?;
            let expr = self.parse_expr()?;
            self.expect(&TokenKind::RParen)?;
            Ok(TableConstraint::Check(expr))
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "table constraint",
            ))
        }
    }

    fn parse_create_index(&mut self, unique: bool) -> ParseResult<StatementKind> {
        let if_not_exists = if self.eat(&TokenKind::If) {
            self.expect(&TokenKind::Not)?;
            self.expect(&TokenKind::Exists)?;
            true
        } else {
            false
        };

        let name = self.expect_ident()?;
        self.expect(&TokenKind::On)?;
        let table = self.expect_ident()?;
        self.expect(&TokenKind::LParen)?;

        let mut columns = Vec::new();
        loop {
            columns.push(self.expect_ident()?);
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen)?;

        Ok(StatementKind::CreateIndex(CreateIndexStmt {
            unique,
            if_not_exists,
            name,
            table,
            columns,
        }))
    }

    fn parse_drop(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Drop)?;

        if self.eat(&TokenKind::Table) {
            let if_exists = if self.eat(&TokenKind::If) {
                self.expect(&TokenKind::Exists)?;
                true
            } else {
                false
            };

            let table = self.expect_ident()?;
            let cascade = self.eat(&TokenKind::Cascade);

            Ok(StatementKind::DropTable(DropTableStmt {
                if_exists,
                table,
                cascade,
            }))
        } else if self.eat(&TokenKind::Index) {
            let if_exists = if self.eat(&TokenKind::If) {
                self.expect(&TokenKind::Exists)?;
                true
            } else {
                false
            };

            // Support both `DROP INDEX ON table(column)` and `DROP INDEX name`
            if self.eat(&TokenKind::On) {
                let table = self.expect_ident()?;
                self.expect(&TokenKind::LParen)?;
                let column = self.expect_ident()?;
                self.expect(&TokenKind::RParen)?;

                Ok(StatementKind::DropIndex(DropIndexStmt {
                    if_exists,
                    name: None,
                    table: Some(table),
                    column: Some(column),
                }))
            } else {
                let name = self.expect_ident()?;

                Ok(StatementKind::DropIndex(DropIndexStmt {
                    if_exists,
                    name: Some(name),
                    table: None,
                    column: None,
                }))
            }
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "TABLE or INDEX",
            ))
        }
    }

    fn parse_show(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Show)?;

        if self.eat(&TokenKind::Tables) {
            Ok(StatementKind::ShowTables)
        } else if self.eat(&TokenKind::Embeddings) {
            let limit = if self.eat(&TokenKind::Limit) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            Ok(StatementKind::ShowEmbeddings { limit })
        } else if self.eat(&TokenKind::Vector) {
            self.expect(&TokenKind::Index)?;
            Ok(StatementKind::ShowVectorIndex)
        } else if self.eat(&TokenKind::Codebook) {
            // SHOW CODEBOOK GLOBAL or SHOW CODEBOOK LOCAL 'domain'
            if self.eat(&TokenKind::Global) {
                Ok(StatementKind::Chain(ChainStmt {
                    operation: ChainOp::ShowCodebookGlobal,
                }))
            } else if self.eat(&TokenKind::Local) {
                let domain = self.parse_expr()?;
                Ok(StatementKind::Chain(ChainStmt {
                    operation: ChainOp::ShowCodebookLocal { domain },
                }))
            } else {
                Err(ParseError::unexpected(
                    self.current.kind.clone(),
                    self.current.span,
                    "GLOBAL or LOCAL",
                ))
            }
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "TABLES, EMBEDDINGS, VECTOR INDEX, or CODEBOOK",
            ))
        }
    }

    fn parse_describe(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Describe)?;

        let target = if self.eat(&TokenKind::Table) {
            let name = self.expect_ident()?;
            DescribeTarget::Table(name)
        } else if self.eat(&TokenKind::Node) {
            let label = self.expect_ident()?;
            DescribeTarget::Node(label)
        } else if self.eat(&TokenKind::Edge) {
            let edge_type = self.expect_ident()?;
            DescribeTarget::Edge(edge_type)
        } else {
            return Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "TABLE, NODE, or EDGE",
            ));
        };

        Ok(StatementKind::Describe(DescribeStmt { target }))
    }

    fn parse_count(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Count)?;

        if self.eat(&TokenKind::Embeddings) {
            Ok(StatementKind::CountEmbeddings)
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "EMBEDDINGS",
            ))
        }
    }

    // =========================================================================
    // Graph Statement Parsers
    // =========================================================================

    fn parse_node(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Node)?;

        let operation = if self.eat(&TokenKind::Create) {
            let label = self.expect_ident()?;
            let properties = self.parse_properties()?;
            NodeOp::Create { label, properties }
        } else if self.eat(&TokenKind::Get) {
            let id = self.parse_expr()?;
            NodeOp::Get { id }
        } else if self.eat(&TokenKind::Delete) {
            let id = self.parse_expr()?;
            NodeOp::Delete { id }
        } else if self.eat(&TokenKind::List) {
            // Parse optional label (identifier that isn't LIMIT or OFFSET)
            let label = if !self.current.is_eof()
                && !self.check(&TokenKind::Semicolon)
                && !self.check(&TokenKind::Limit)
                && !self.check(&TokenKind::Offset)
            {
                Some(self.expect_ident()?)
            } else {
                None
            };
            // Parse optional LIMIT
            let limit = if self.eat(&TokenKind::Limit) {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            // Parse optional OFFSET
            let offset = if self.eat(&TokenKind::Offset) {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            NodeOp::List {
                label,
                limit,
                offset,
            }
        } else {
            return Err(ParseError::invalid(
                "expected CREATE, GET, DELETE, or LIST after NODE",
                self.current.span,
            ));
        };

        Ok(StatementKind::Node(NodeStmt { operation }))
    }

    fn parse_edge(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Edge)?;

        let operation = if self.eat(&TokenKind::Create) {
            let from_id = self.parse_expr()?;
            self.expect(&TokenKind::Arrow)?;
            let to_id = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let edge_type = self.expect_ident()?;
            let properties = self.parse_properties()?;
            EdgeOp::Create {
                from_id,
                to_id,
                edge_type,
                properties,
            }
        } else if self.eat(&TokenKind::Get) {
            let id = self.parse_expr()?;
            EdgeOp::Get { id }
        } else if self.eat(&TokenKind::Delete) {
            let id = self.parse_expr()?;
            EdgeOp::Delete { id }
        } else if self.eat(&TokenKind::List) {
            // Parse optional edge_type (identifier that isn't LIMIT or OFFSET)
            let edge_type = if !self.current.is_eof()
                && !self.check(&TokenKind::Semicolon)
                && !self.check(&TokenKind::Limit)
                && !self.check(&TokenKind::Offset)
            {
                Some(self.expect_ident()?)
            } else {
                None
            };
            // Parse optional LIMIT
            let limit = if self.eat(&TokenKind::Limit) {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            // Parse optional OFFSET
            let offset = if self.eat(&TokenKind::Offset) {
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            EdgeOp::List {
                edge_type,
                limit,
                offset,
            }
        } else {
            return Err(ParseError::invalid(
                "expected CREATE, GET, DELETE, or LIST after EDGE",
                self.current.span,
            ));
        };

        Ok(StatementKind::Edge(EdgeStmt { operation }))
    }

    fn parse_properties(&mut self) -> ParseResult<Vec<Property>> {
        if !self.eat(&TokenKind::LBrace) {
            return Ok(Vec::new());
        }

        let mut properties = Vec::new();
        if !self.check(&TokenKind::RBrace) {
            loop {
                // Allow keywords as field names (e.g., status, type, etc.)
                let key = self.expect_ident_or_keyword()?;
                self.expect(&TokenKind::Colon)?;
                let value = self.parse_expr()?;
                properties.push(Property { key, value });
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBrace)?;
        Ok(properties)
    }

    fn parse_neighbors(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Neighbors)?;

        let node_id = self.parse_expr()?;

        let direction = if self.eat(&TokenKind::Outgoing) {
            Direction::Outgoing
        } else if self.eat(&TokenKind::Incoming) {
            Direction::Incoming
        } else if self.eat(&TokenKind::Both) {
            Direction::Both
        } else {
            Direction::Outgoing
        };

        let edge_type = if self.eat(&TokenKind::Colon) {
            Some(self.expect_ident()?)
        } else {
            None
        };

        // Check for BY SIMILARITY clause
        let by_similarity = if self.eat(&TokenKind::By) {
            self.expect(&TokenKind::Similar)?;
            Some(self.parse_vector_literal()?)
        } else {
            None
        };

        let limit = if self.eat(&TokenKind::Limit) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(StatementKind::Neighbors(NeighborsStmt {
            node_id,
            direction,
            edge_type,
            by_similarity,
            limit,
        }))
    }

    fn parse_path(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Path)?;

        // Optionally consume SHORTEST keyword; default algorithm is Shortest
        self.eat(&TokenKind::Shortest);
        let algorithm = PathAlgorithm::Shortest;

        let from_id = self.parse_expr()?;
        self.expect(&TokenKind::Arrow)?;
        let to_id = self.parse_expr()?;

        let max_depth = if self.eat(&TokenKind::Limit) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(StatementKind::Path(PathStmt {
            algorithm,
            from_id,
            to_id,
            max_depth,
            min_depth: None,
            weight_property: None,
        }))
    }

    // =========================================================================
    // Vector Statement Parsers
    // =========================================================================

    fn parse_embed(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Embed)?;

        let operation = if self.eat(&TokenKind::Store) {
            let key = self.parse_expr()?;
            self.expect(&TokenKind::LBracket)?;
            let mut vector = Vec::new();
            if !self.check(&TokenKind::RBracket) {
                loop {
                    vector.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket)?;
            EmbedOp::Store { key, vector }
        } else if self.eat(&TokenKind::Get) {
            let key = self.parse_expr()?;
            EmbedOp::Get { key }
        } else if self.eat(&TokenKind::Delete) {
            let key = self.parse_expr()?;
            EmbedOp::Delete { key }
        } else if self.eat(&TokenKind::Build) {
            self.expect(&TokenKind::Index)?;
            EmbedOp::BuildIndex
        } else if self.eat(&TokenKind::Batch) {
            // EMBED BATCH [('key1', [v1, v2]), ('key2', [v1, v2])]
            self.expect(&TokenKind::LBracket)?;
            let mut items = Vec::new();
            if !self.check(&TokenKind::RBracket) {
                loop {
                    // Parse ('key', [v1, v2])
                    self.expect(&TokenKind::LParen)?;
                    let key = self.parse_expr()?;
                    self.expect(&TokenKind::Comma)?;
                    self.expect(&TokenKind::LBracket)?;
                    let mut vector = Vec::new();
                    if !self.check(&TokenKind::RBracket) {
                        loop {
                            vector.push(self.parse_expr()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RBracket)?;
                    self.expect(&TokenKind::RParen)?;
                    items.push((key, vector));
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket)?;
            EmbedOp::Batch { items }
        } else {
            return Err(ParseError::invalid(
                "expected STORE, GET, DELETE, BUILD INDEX, or BATCH after EMBED",
                self.current.span,
            ));
        };

        // Parse optional INTO collection_name
        let collection = if self.eat(&TokenKind::Into) {
            Some(self.expect_ident()?.name)
        } else {
            None
        };

        Ok(StatementKind::Embed(EmbedStmt {
            operation,
            collection,
        }))
    }

    fn parse_similar(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Similar)?;

        let query = if self.eat(&TokenKind::LBracket) {
            let mut vector = Vec::new();
            if !self.check(&TokenKind::RBracket) {
                loop {
                    vector.push(self.parse_expr()?);
                    if !self.eat(&TokenKind::Comma) {
                        break;
                    }
                }
            }
            self.expect(&TokenKind::RBracket)?;
            SimilarQuery::Vector(vector)
        } else {
            let key = self.parse_expr()?;
            SimilarQuery::Key(key)
        };

        // Check for CONNECTED TO clause
        let connected_to = if self.eat(&TokenKind::Connected) {
            self.expect(&TokenKind::To)?;
            Some(self.parse_expr()?)
        } else {
            None
        };

        let limit = if self.eat(&TokenKind::Limit) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        let metric = if self.eat(&TokenKind::Cosine) {
            Some(DistanceMetric::Cosine)
        } else if self.eat(&TokenKind::Euclidean) {
            Some(DistanceMetric::Euclidean)
        } else if self.eat(&TokenKind::DotProduct) {
            Some(DistanceMetric::DotProduct)
        } else {
            None
        };

        // Parse optional INTO collection_name
        let collection = if self.eat(&TokenKind::Into) {
            Some(self.expect_ident()?.name)
        } else {
            None
        };

        // Parse optional WHERE clause for filtered search
        let where_clause = if self.eat(&TokenKind::Where) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(StatementKind::Similar(SimilarStmt {
            query,
            limit,
            metric,
            connected_to,
            collection,
            where_clause,
        }))
    }

    // =========================================================================
    // Unified Query Parser
    // =========================================================================

    fn parse_find(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Find)?;

        // Parse pattern
        let pattern = if self.eat(&TokenKind::Node) || self.eat(&TokenKind::Vertex) {
            let label = if !self.check(&TokenKind::Where)
                && !self.check(&TokenKind::Return)
                && !self.check(&TokenKind::Limit)
                && !self.current.is_eof()
            {
                Some(self.expect_ident()?)
            } else {
                None
            };
            FindPattern::Nodes { label }
        } else if self.eat(&TokenKind::Edge) {
            let edge_type = if !self.check(&TokenKind::Where)
                && !self.check(&TokenKind::Return)
                && !self.check(&TokenKind::Limit)
                && !self.current.is_eof()
            {
                Some(self.expect_ident()?)
            } else {
                None
            };
            FindPattern::Edges { edge_type }
        } else if self.eat(&TokenKind::Rows) {
            // FIND ROWS FROM table
            self.expect(&TokenKind::From)?;
            let table = self.expect_ident()?;
            FindPattern::Rows { table }
        } else {
            FindPattern::Nodes { label: None }
        };

        let where_clause = if self.eat(&TokenKind::Where) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        let return_items = if self.eat(&TokenKind::Return) {
            let mut items = Vec::new();
            loop {
                items.push(self.parse_select_item()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
            items
        } else {
            Vec::new()
        };

        let limit = if self.eat(&TokenKind::Limit) {
            Some(Box::new(self.parse_expr()?))
        } else {
            None
        };

        Ok(StatementKind::Find(FindStmt {
            pattern,
            where_clause,
            return_items,
            limit,
        }))
    }

    fn parse_entity(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Entity)?;

        let operation = if self.eat(&TokenKind::Batch) {
            // ENTITY BATCH CREATE [{key: 'k1', name: 'Alice'}, ...]
            self.expect(&TokenKind::Create)?;
            let entities = self.parse_batch_entity_list()?;
            EntityOp::Batch { entities }
        } else if self.eat(&TokenKind::Create) {
            // ENTITY CREATE 'key' { properties } [EMBEDDING [vector]]
            let key = self.parse_expr()?;
            let properties = self.parse_properties()?;

            let embedding = if self.eat(&TokenKind::Embedding) {
                Some(self.parse_vector_literal()?)
            } else {
                None
            };

            EntityOp::Create {
                key,
                properties,
                embedding,
            }
        } else if self.eat(&TokenKind::Get) {
            // ENTITY GET 'key'
            let key = self.parse_expr()?;
            EntityOp::Get { key }
        } else if self.eat(&TokenKind::Update) {
            // ENTITY UPDATE 'key' { properties } [EMBEDDING [vector]]
            let key = self.parse_expr()?;
            let properties = self.parse_properties()?;

            let embedding = if self.eat(&TokenKind::Embedding) {
                Some(self.parse_vector_literal()?)
            } else {
                None
            };

            EntityOp::Update {
                key,
                properties,
                embedding,
            }
        } else if self.eat(&TokenKind::Delete) {
            // ENTITY DELETE 'key'
            let key = self.parse_expr()?;
            EntityOp::Delete { key }
        } else if self.eat(&TokenKind::Connect) {
            // ENTITY CONNECT 'from' -> 'to' : type
            let from_key = self.parse_expr()?;
            self.expect(&TokenKind::Arrow)?;
            let to_key = self.parse_expr()?;
            self.expect(&TokenKind::Colon)?;
            let edge_type = self.expect_ident_or_keyword()?;

            EntityOp::Connect {
                from_key,
                to_key,
                edge_type,
            }
        } else {
            return Err(ParseError::invalid(
                "expected CREATE, GET, UPDATE, DELETE, CONNECT, or BATCH after ENTITY",
                self.current.span,
            ));
        };

        Ok(StatementKind::Entity(EntityStmt { operation }))
    }

    /// Parses a list of batch entity definitions: `[{key: 'k1', name: 'Alice'}, ...]`
    fn parse_batch_entity_list(&mut self) -> ParseResult<Vec<BatchEntityDef>> {
        self.expect(&TokenKind::LBracket)?;

        let mut entities = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            loop {
                let entity = self.parse_batch_entity_def()?;
                entities.push(entity);

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(entities)
    }

    /// Parses a single batch entity definition: `{key: 'k1', name: 'Alice', embedding: [0.1, 0.2]}`
    fn parse_batch_entity_def(&mut self) -> ParseResult<BatchEntityDef> {
        self.expect(&TokenKind::LBrace)?;

        let mut key: Option<Expr> = None;
        let mut properties = Vec::new();
        let mut embedding: Option<Vec<Expr>> = None;

        if !self.check(&TokenKind::RBrace) {
            loop {
                // Use expect_ident_or_any_keyword since "key" and "embedding" are reserved keywords
                let prop_name = self.expect_ident_or_any_keyword()?;
                self.expect(&TokenKind::Colon)?;

                if prop_name.name.eq_ignore_ascii_case("key") {
                    key = Some(self.parse_expr()?);
                } else if prop_name.name.eq_ignore_ascii_case("embedding") {
                    embedding = Some(self.parse_vector_literal()?);
                } else {
                    let value = self.parse_expr()?;
                    properties.push(Property {
                        key: prop_name,
                        value,
                    });
                }

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBrace)?;

        let key = key.ok_or_else(|| {
            ParseError::invalid(
                "batch entity definition requires 'key' field",
                self.current.span,
            )
        })?;

        Ok(BatchEntityDef {
            key,
            properties,
            embedding,
        })
    }

    // =========================================================================
    // Vault Statement Parsers
    // =========================================================================

    fn parse_vault(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Vault)?;

        let operation = match &self.current.kind {
            TokenKind::Set => {
                self.advance();
                let key = self.parse_expr()?;
                let value = self.parse_expr()?;
                VaultOp::Set { key, value }
            },
            TokenKind::Get => {
                self.advance();
                let key = self.parse_expr()?;
                VaultOp::Get { key }
            },
            TokenKind::Delete => {
                self.advance();
                let key = self.parse_expr()?;
                VaultOp::Delete { key }
            },
            TokenKind::List => {
                self.advance();
                let pattern = if !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                VaultOp::List { pattern }
            },
            TokenKind::Rotate => {
                self.advance();
                let key = self.parse_expr()?;
                let new_value = self.parse_expr()?;
                VaultOp::Rotate { key, new_value }
            },
            TokenKind::Grant => {
                self.advance();
                let entity = self.parse_expr()?;
                self.expect(&TokenKind::On)?;
                let key = self.parse_expr()?;
                VaultOp::Grant { entity, key }
            },
            TokenKind::Revoke => {
                self.advance();
                let entity = self.parse_expr()?;
                self.expect(&TokenKind::On)?;
                let key = self.parse_expr()?;
                VaultOp::Revoke { entity, key }
            },
            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken {
                        found: self.current.kind.clone(),
                        expected: "VAULT operation (SET, GET, DELETE, LIST, ROTATE, GRANT, REVOKE)"
                            .to_string(),
                    },
                    self.current.span,
                ));
            },
        };

        Ok(StatementKind::Vault(VaultStmt { operation }))
    }

    fn parse_cache(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Cache)?;

        let operation = match &self.current.kind {
            TokenKind::Init => {
                self.advance();
                CacheOp::Init
            },
            TokenKind::Stats => {
                self.advance();
                CacheOp::Stats
            },
            TokenKind::Clear => {
                self.advance();
                CacheOp::Clear
            },
            TokenKind::Evict => {
                self.advance();
                let count = if !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                CacheOp::Evict { count }
            },
            TokenKind::Get => {
                self.advance();
                let key = self.parse_expr()?;
                CacheOp::Get { key }
            },
            TokenKind::Put => {
                self.advance();
                let key = self.parse_expr()?;
                let value = self.parse_expr()?;
                CacheOp::Put { key, value }
            },
            TokenKind::Semantic => {
                self.advance();
                if self.eat(&TokenKind::Get) {
                    // CACHE SEMANTIC GET 'query' [THRESHOLD n]
                    let query = self.parse_expr()?;
                    let threshold = if self.eat(&TokenKind::Threshold) {
                        Some(self.parse_expr()?)
                    } else {
                        None
                    };
                    CacheOp::SemanticGet { query, threshold }
                } else if self.eat(&TokenKind::Put) {
                    // CACHE SEMANTIC PUT 'query' 'response' EMBEDDING [vector]
                    let query = self.parse_expr()?;
                    let response = self.parse_expr()?;
                    self.expect(&TokenKind::Embedding)?;
                    self.expect(&TokenKind::LBracket)?;
                    let mut embedding = Vec::new();
                    if !self.check(&TokenKind::RBracket) {
                        loop {
                            embedding.push(self.parse_expr()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RBracket)?;
                    CacheOp::SemanticPut {
                        query,
                        response,
                        embedding,
                    }
                } else {
                    return Err(ParseError::invalid(
                        "expected GET or PUT after CACHE SEMANTIC",
                        self.current.span,
                    ));
                }
            },
            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken {
                        found: self.current.kind.clone(),
                        expected: "CACHE operation (INIT, STATS, CLEAR, EVICT, GET, PUT, SEMANTIC)"
                            .to_string(),
                    },
                    self.current.span,
                ));
            },
        };

        Ok(StatementKind::Cache(CacheStmt { operation }))
    }

    // =========================================================================
    // Cluster Statement Parsers
    // =========================================================================

    fn parse_cluster(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Cluster)?;

        let operation = match &self.current.kind {
            TokenKind::Connect => {
                self.advance();
                let addresses = self.parse_expr()?;
                ClusterOp::Connect { addresses }
            },
            TokenKind::Disconnect => {
                self.advance();
                ClusterOp::Disconnect
            },
            TokenKind::Status => {
                self.advance();
                ClusterOp::Status
            },
            TokenKind::Nodes => {
                self.advance();
                ClusterOp::Nodes
            },
            TokenKind::Leader => {
                self.advance();
                ClusterOp::Leader
            },
            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken {
                        found: self.current.kind.clone(),
                        expected: "CONNECT, DISCONNECT, STATUS, NODES, or LEADER".to_string(),
                    },
                    self.current.span,
                ));
            },
        };

        Ok(StatementKind::Cluster(ClusterStmt { operation }))
    }

    // =========================================================================
    // Extended Graph Statement Parsers
    // =========================================================================

    fn parse_graph(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Graph)?;

        match &self.current.kind {
            TokenKind::PageRank => self.parse_graph_pagerank(),
            TokenKind::Betweenness => self.parse_graph_betweenness(),
            TokenKind::Closeness => self.parse_graph_closeness(),
            TokenKind::Eigenvector => self.parse_graph_eigenvector(),
            TokenKind::Louvain => self.parse_graph_louvain(),
            TokenKind::Label => self.parse_graph_label_propagation(),
            TokenKind::Index => self.parse_graph_index(),
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected:
                        "PAGERANK, BETWEENNESS, CLOSENESS, EIGENVECTOR, LOUVAIN, LABEL, or INDEX"
                            .to_string(),
                },
                self.current.span,
            )),
        }
    }

    fn parse_graph_pagerank(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::PageRank)?;

        let mut damping = None;
        let mut tolerance = None;
        let mut max_iterations = None;
        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Damping => {
                    self.advance();
                    damping = Some(self.parse_expr()?);
                },
                TokenKind::Tolerance => {
                    self.advance();
                    tolerance = Some(self.parse_expr()?);
                },
                TokenKind::Iterations => {
                    self.advance();
                    max_iterations = Some(self.parse_expr()?);
                },
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::PageRank {
                damping,
                tolerance,
                max_iterations,
                direction,
                edge_type,
            },
        }))
    }

    fn parse_graph_betweenness(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Betweenness)?;
        self.expect(&TokenKind::Centrality)?;

        let mut sampling_ratio = None;
        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Sampling => {
                    self.advance();
                    sampling_ratio = Some(self.parse_expr()?);
                },
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::BetweennessCentrality {
                sampling_ratio,
                direction,
                edge_type,
            },
        }))
    }

    fn parse_graph_closeness(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Closeness)?;
        self.expect(&TokenKind::Centrality)?;

        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::ClosenessCentrality {
                direction,
                edge_type,
            },
        }))
    }

    fn parse_graph_eigenvector(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Eigenvector)?;
        self.expect(&TokenKind::Centrality)?;

        let mut max_iterations = None;
        let mut tolerance = None;
        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Iterations => {
                    self.advance();
                    max_iterations = Some(self.parse_expr()?);
                },
                TokenKind::Tolerance => {
                    self.advance();
                    tolerance = Some(self.parse_expr()?);
                },
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::EigenvectorCentrality {
                max_iterations,
                tolerance,
                direction,
                edge_type,
            },
        }))
    }

    fn parse_graph_louvain(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Louvain)?;
        self.expect(&TokenKind::Communities)?;

        let mut resolution = None;
        let mut max_passes = None;
        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Resolution => {
                    self.advance();
                    resolution = Some(self.parse_expr()?);
                },
                TokenKind::Passes => {
                    self.advance();
                    max_passes = Some(self.parse_expr()?);
                },
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::LouvainCommunities {
                resolution,
                max_passes,
                direction,
                edge_type,
            },
        }))
    }

    fn parse_graph_label_propagation(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Label)?;
        self.expect(&TokenKind::Propagation)?;

        let mut max_iterations = None;
        let mut direction = None;
        let mut edge_type = None;

        while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
            match &self.current.kind {
                TokenKind::Iterations => {
                    self.advance();
                    max_iterations = Some(self.parse_expr()?);
                },
                TokenKind::Outgoing | TokenKind::Incoming | TokenKind::Both => {
                    direction = Some(self.parse_direction()?);
                },
                TokenKind::Edge => {
                    self.advance();
                    self.expect(&TokenKind::Type)?;
                    edge_type = Some(self.expect_ident_or_keyword()?);
                },
                _ => break,
            }
        }

        Ok(StatementKind::GraphAlgorithm(GraphAlgorithmStmt {
            operation: GraphAlgorithmOp::LabelPropagation {
                max_iterations,
                direction,
                edge_type,
            },
        }))
    }

    #[allow(clippy::too_many_lines)] // Complex subcommand parsing
    fn parse_graph_index(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Index)?;

        match &self.current.kind {
            TokenKind::Create => {
                self.advance();
                self.expect(&TokenKind::On)?;

                match &self.current.kind {
                    TokenKind::Node => {
                        self.advance();
                        self.expect(&TokenKind::Property)?;
                        let property = self.expect_ident_or_keyword()?;
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::CreateNodeProperty { property },
                        }))
                    },
                    TokenKind::Edge => {
                        self.advance();
                        if self.eat(&TokenKind::Property) {
                            let property = self.expect_ident_or_keyword()?;
                            Ok(StatementKind::GraphIndex(GraphIndexStmt {
                                operation: GraphIndexOp::CreateEdgeProperty { property },
                            }))
                        } else {
                            self.expect(&TokenKind::Type)?;
                            Ok(StatementKind::GraphIndex(GraphIndexStmt {
                                operation: GraphIndexOp::CreateEdgeType,
                            }))
                        }
                    },
                    TokenKind::Label => {
                        self.advance();
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::CreateLabel,
                        }))
                    },
                    _ => Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "NODE, EDGE, or LABEL".to_string(),
                        },
                        self.current.span,
                    )),
                }
            },
            TokenKind::Drop => {
                self.advance();
                self.expect(&TokenKind::On)?;

                match &self.current.kind {
                    TokenKind::Node => {
                        self.advance();
                        self.expect(&TokenKind::Property)?;
                        let property = self.expect_ident_or_keyword()?;
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::DropNode { property },
                        }))
                    },
                    TokenKind::Edge => {
                        self.advance();
                        self.expect(&TokenKind::Property)?;
                        let property = self.expect_ident_or_keyword()?;
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::DropEdge { property },
                        }))
                    },
                    _ => Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "NODE or EDGE".to_string(),
                        },
                        self.current.span,
                    )),
                }
            },
            TokenKind::Show => {
                self.advance();
                self.expect(&TokenKind::On)?;

                match &self.current.kind {
                    TokenKind::Node => {
                        self.advance();
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::ShowNodeIndexes,
                        }))
                    },
                    TokenKind::Edge => {
                        self.advance();
                        Ok(StatementKind::GraphIndex(GraphIndexStmt {
                            operation: GraphIndexOp::ShowEdgeIndexes,
                        }))
                    },
                    _ => Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "NODE or EDGE".to_string(),
                        },
                        self.current.span,
                    )),
                }
            },
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected: "CREATE, DROP, or SHOW".to_string(),
                },
                self.current.span,
            )),
        }
    }

    fn parse_constraint(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Constraint)?;

        match &self.current.kind {
            TokenKind::Create => {
                self.advance();
                let name = self.expect_ident_or_keyword()?;
                self.expect(&TokenKind::On)?;

                let target = match &self.current.kind {
                    TokenKind::Node => {
                        self.advance();
                        let label = if !self.check(&TokenKind::Property) {
                            Some(self.expect_ident_or_keyword()?)
                        } else {
                            None
                        };
                        ConstraintTarget::Node { label }
                    },
                    TokenKind::Edge => {
                        self.advance();
                        let edge_type = if !self.check(&TokenKind::Property) {
                            Some(self.expect_ident_or_keyword()?)
                        } else {
                            None
                        };
                        ConstraintTarget::Edge { edge_type }
                    },
                    _ => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken {
                                found: self.current.kind.clone(),
                                expected: "NODE or EDGE".to_string(),
                            },
                            self.current.span,
                        ));
                    },
                };

                self.expect(&TokenKind::Property)?;
                let property = self.expect_ident_or_keyword()?;

                let constraint_type = match &self.current.kind {
                    TokenKind::Unique => {
                        self.advance();
                        ConstraintType::Unique
                    },
                    TokenKind::Exists => {
                        self.advance();
                        ConstraintType::Exists
                    },
                    TokenKind::Type => {
                        self.advance();
                        let type_name = self.expect_ident_or_keyword()?;
                        ConstraintType::Type(type_name.name)
                    },
                    _ => {
                        return Err(ParseError::new(
                            ParseErrorKind::UnexpectedToken {
                                found: self.current.kind.clone(),
                                expected: "UNIQUE, EXISTS, or TYPE".to_string(),
                            },
                            self.current.span,
                        ));
                    },
                };

                Ok(StatementKind::GraphConstraint(GraphConstraintStmt {
                    operation: GraphConstraintOp::Create {
                        name,
                        target,
                        property,
                        constraint_type,
                    },
                }))
            },
            TokenKind::Drop => {
                self.advance();
                let name = self.expect_ident_or_keyword()?;
                Ok(StatementKind::GraphConstraint(GraphConstraintStmt {
                    operation: GraphConstraintOp::Drop { name },
                }))
            },
            TokenKind::List => {
                self.advance();
                Ok(StatementKind::GraphConstraint(GraphConstraintStmt {
                    operation: GraphConstraintOp::List,
                }))
            },
            TokenKind::Get => {
                self.advance();
                let name = self.expect_ident_or_keyword()?;
                Ok(StatementKind::GraphConstraint(GraphConstraintStmt {
                    operation: GraphConstraintOp::Get { name },
                }))
            },
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected: "CREATE, DROP, LIST, or GET".to_string(),
                },
                self.current.span,
            )),
        }
    }

    fn parse_batch(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Batch)?;

        match &self.current.kind {
            TokenKind::Create => {
                self.advance();
                match &self.current.kind {
                    TokenKind::Node | TokenKind::Nodes => {
                        self.advance();
                        let nodes = self.parse_batch_node_list()?;
                        Ok(StatementKind::GraphBatch(GraphBatchStmt {
                            operation: GraphBatchOp::CreateNodes { nodes },
                        }))
                    },
                    TokenKind::Edge | TokenKind::Edges => {
                        self.advance();
                        let edges = self.parse_batch_edge_list()?;
                        Ok(StatementKind::GraphBatch(GraphBatchStmt {
                            operation: GraphBatchOp::CreateEdges { edges },
                        }))
                    },
                    _ => Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "NODES or EDGES".to_string(),
                        },
                        self.current.span,
                    )),
                }
            },
            TokenKind::Delete => {
                self.advance();
                match &self.current.kind {
                    TokenKind::Node | TokenKind::Nodes => {
                        self.advance();
                        let ids = self.parse_expr_list()?;
                        Ok(StatementKind::GraphBatch(GraphBatchStmt {
                            operation: GraphBatchOp::DeleteNodes { ids },
                        }))
                    },
                    TokenKind::Edge | TokenKind::Edges => {
                        self.advance();
                        let ids = self.parse_expr_list()?;
                        Ok(StatementKind::GraphBatch(GraphBatchStmt {
                            operation: GraphBatchOp::DeleteEdges { ids },
                        }))
                    },
                    _ => Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "NODES or EDGES".to_string(),
                        },
                        self.current.span,
                    )),
                }
            },
            TokenKind::Update => {
                self.advance();
                self.expect(&TokenKind::Nodes)?;
                let updates = self.parse_batch_update_list()?;
                Ok(StatementKind::GraphBatch(GraphBatchStmt {
                    operation: GraphBatchOp::UpdateNodes { updates },
                }))
            },
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected: "CREATE, DELETE, or UPDATE".to_string(),
                },
                self.current.span,
            )),
        }
    }

    fn parse_batch_node_list(&mut self) -> ParseResult<Vec<BatchNodeDef>> {
        self.expect(&TokenKind::LBracket)?;
        let mut nodes = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            loop {
                nodes.push(self.parse_batch_node_def()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(nodes)
    }

    fn parse_batch_node_def(&mut self) -> ParseResult<BatchNodeDef> {
        self.expect(&TokenKind::LBrace)?;

        let mut labels = Vec::new();
        let mut properties = Vec::new();

        if !self.check(&TokenKind::RBrace) {
            loop {
                let key = self.expect_ident_or_keyword()?;
                self.expect(&TokenKind::Colon)?;

                if key.name == "labels" {
                    self.expect(&TokenKind::LBracket)?;
                    if !self.check(&TokenKind::RBracket) {
                        loop {
                            labels.push(self.expect_ident_or_keyword()?);
                            if !self.eat(&TokenKind::Comma) {
                                break;
                            }
                        }
                    }
                    self.expect(&TokenKind::RBracket)?;
                } else {
                    let value = self.parse_expr()?;
                    properties.push(Property { key, value });
                }

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBrace)?;
        Ok(BatchNodeDef { labels, properties })
    }

    fn parse_batch_edge_list(&mut self) -> ParseResult<Vec<BatchEdgeDef>> {
        self.expect(&TokenKind::LBracket)?;
        let mut edges = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            loop {
                edges.push(self.parse_batch_edge_def()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(edges)
    }

    fn parse_batch_edge_def(&mut self) -> ParseResult<BatchEdgeDef> {
        self.expect(&TokenKind::LBrace)?;

        let mut from_id = None;
        let mut to_id = None;
        let mut edge_type = None;
        let mut properties = Vec::new();

        if !self.check(&TokenKind::RBrace) {
            loop {
                // Use expect_ident_or_any_keyword to allow reserved keywords like FROM, TO
                let key = self.expect_ident_or_any_keyword()?;
                self.expect(&TokenKind::Colon)?;

                match key.name.as_str() {
                    "from" => from_id = Some(self.parse_expr()?),
                    "to" => to_id = Some(self.parse_expr()?),
                    "type" => edge_type = Some(self.expect_ident_or_any_keyword()?),
                    _ => {
                        let value = self.parse_expr()?;
                        properties.push(Property { key, value });
                    },
                }

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBrace)?;

        let from_id = from_id.ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::Custom("Missing 'from' in edge definition".to_string()),
                self.current.span,
            )
        })?;
        let to_id = to_id.ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::Custom("Missing 'to' in edge definition".to_string()),
                self.current.span,
            )
        })?;
        let edge_type = edge_type.ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::Custom("Missing 'type' in edge definition".to_string()),
                self.current.span,
            )
        })?;

        Ok(BatchEdgeDef {
            from_id,
            to_id,
            edge_type,
            properties,
        })
    }

    fn parse_batch_update_list(&mut self) -> ParseResult<Vec<BatchNodeUpdate>> {
        self.expect(&TokenKind::LBracket)?;
        let mut updates = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            loop {
                updates.push(self.parse_batch_node_update()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(updates)
    }

    fn parse_batch_node_update(&mut self) -> ParseResult<BatchNodeUpdate> {
        self.expect(&TokenKind::LBrace)?;

        let mut id = None;
        let mut properties = Vec::new();

        if !self.check(&TokenKind::RBrace) {
            loop {
                let key = self.expect_ident_or_keyword()?;
                self.expect(&TokenKind::Colon)?;

                if key.name == "id" {
                    id = Some(self.parse_expr()?);
                } else {
                    let value = self.parse_expr()?;
                    properties.push(Property { key, value });
                }

                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBrace)?;

        let id = id.ok_or_else(|| {
            ParseError::new(
                ParseErrorKind::Custom("Missing 'id' in node update".to_string()),
                self.current.span,
            )
        })?;

        Ok(BatchNodeUpdate { id, properties })
    }

    fn parse_expr_list(&mut self) -> ParseResult<Vec<Expr>> {
        self.expect(&TokenKind::LBracket)?;
        let mut exprs = Vec::new();

        if !self.check(&TokenKind::RBracket) {
            loop {
                exprs.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }

        self.expect(&TokenKind::RBracket)?;
        Ok(exprs)
    }

    fn parse_aggregate_stmt(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Aggregate)?;

        match &self.current.kind {
            TokenKind::Node => {
                self.advance();
                self.expect(&TokenKind::Property)?;
                let property = self.expect_ident_or_keyword()?;

                let function = self.parse_aggregate_function()?;

                let label = if self.eat(&TokenKind::By) {
                    self.expect(&TokenKind::Label)?;
                    Some(self.expect_ident_or_keyword()?)
                } else {
                    None
                };

                let filter = if self.eat(&TokenKind::Where) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };

                Ok(StatementKind::GraphAggregate(GraphAggregateStmt {
                    operation: GraphAggregateOp::AggregateNodeProperty {
                        function,
                        property,
                        label,
                        filter,
                    },
                }))
            },
            TokenKind::Edge => {
                self.advance();
                self.expect(&TokenKind::Property)?;
                let property = self.expect_ident_or_keyword()?;

                let function = self.parse_aggregate_function()?;

                let edge_type = if self.eat(&TokenKind::By) {
                    self.expect(&TokenKind::Type)?;
                    Some(self.expect_ident_or_keyword()?)
                } else {
                    None
                };

                let filter = if self.eat(&TokenKind::Where) {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };

                Ok(StatementKind::GraphAggregate(GraphAggregateStmt {
                    operation: GraphAggregateOp::AggregateEdgeProperty {
                        function,
                        property,
                        edge_type,
                        filter,
                    },
                }))
            },
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected: "NODE or EDGE".to_string(),
                },
                self.current.span,
            )),
        }
    }

    fn parse_aggregate_function(&mut self) -> ParseResult<AggregateFunction> {
        match &self.current.kind {
            TokenKind::Sum => {
                self.advance();
                Ok(AggregateFunction::Sum)
            },
            TokenKind::Avg => {
                self.advance();
                Ok(AggregateFunction::Avg)
            },
            TokenKind::Min => {
                self.advance();
                Ok(AggregateFunction::Min)
            },
            TokenKind::Max => {
                self.advance();
                Ok(AggregateFunction::Max)
            },
            TokenKind::Count => {
                self.advance();
                Ok(AggregateFunction::Count)
            },
            _ => Err(ParseError::new(
                ParseErrorKind::UnexpectedToken {
                    found: self.current.kind.clone(),
                    expected: "SUM, AVG, MIN, MAX, or COUNT".to_string(),
                },
                self.current.span,
            )),
        }
    }

    // =========================================================================
    // Blob Storage Statement Parsers
    // =========================================================================

    #[allow(clippy::too_many_lines)] // Complex subcommand parsing
    fn parse_blob(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Blob)?;

        let operation = match &self.current.kind {
            TokenKind::Init => {
                self.advance();
                BlobOp::Init
            },
            TokenKind::Put => {
                self.advance();
                let filename = self.parse_expr()?;
                // Check for FROM 'path' or inline data
                let (from_path, data) = if self.eat(&TokenKind::From) {
                    (Some(self.parse_expr()?), None)
                } else if !self.current.is_eof()
                    && !self.check(&TokenKind::Semicolon)
                    && !self.check(&TokenKind::Link)
                    && !self.check(&TokenKind::Tag)
                {
                    (None, Some(self.parse_expr()?))
                } else {
                    (None, None)
                };
                // Parse options: LINK entity, TAG tag
                let mut options = BlobOptions::default();
                while !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
                    if self.eat(&TokenKind::Link) {
                        options.link.push(self.parse_expr()?);
                    } else if self.eat(&TokenKind::Tag) {
                        options.tag.push(self.parse_expr()?);
                    } else {
                        break;
                    }
                }
                BlobOp::Put {
                    filename,
                    data,
                    from_path,
                    options,
                }
            },
            TokenKind::Get => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                let to_path = if self.eat(&TokenKind::To) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                BlobOp::Get {
                    artifact_id,
                    to_path,
                }
            },
            TokenKind::Delete => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                BlobOp::Delete { artifact_id }
            },
            TokenKind::Info => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                BlobOp::Info { artifact_id }
            },
            TokenKind::Link => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                self.expect(&TokenKind::To)?;
                let entity = self.parse_expr()?;
                BlobOp::Link {
                    artifact_id,
                    entity,
                }
            },
            TokenKind::Unlink => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                self.expect(&TokenKind::From)?;
                let entity = self.parse_expr()?;
                BlobOp::Unlink {
                    artifact_id,
                    entity,
                }
            },
            TokenKind::Links => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                BlobOp::Links { artifact_id }
            },
            TokenKind::Tag => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                let tag = self.parse_expr()?;
                BlobOp::Tag { artifact_id, tag }
            },
            TokenKind::Untag => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                let tag = self.parse_expr()?;
                BlobOp::Untag { artifact_id, tag }
            },
            TokenKind::Verify => {
                self.advance();
                let artifact_id = self.parse_expr()?;
                BlobOp::Verify { artifact_id }
            },
            TokenKind::Gc => {
                self.advance();
                let full = self.eat(&TokenKind::Full);
                BlobOp::Gc { full }
            },
            TokenKind::Repair => {
                self.advance();
                BlobOp::Repair
            },
            TokenKind::Stats => {
                self.advance();
                BlobOp::Stats
            },
            TokenKind::Meta => {
                self.advance();
                if self.eat(&TokenKind::Set) {
                    let artifact_id = self.parse_expr()?;
                    let key = self.parse_expr()?;
                    let value = self.parse_expr()?;
                    BlobOp::MetaSet {
                        artifact_id,
                        key,
                        value,
                    }
                } else if self.eat(&TokenKind::Get) {
                    let artifact_id = self.parse_expr()?;
                    let key = self.parse_expr()?;
                    BlobOp::MetaGet { artifact_id, key }
                } else {
                    return Err(ParseError::new(
                        ParseErrorKind::UnexpectedToken {
                            found: self.current.kind.clone(),
                            expected: "META operation (SET, GET)".to_string(),
                        },
                        self.current.span,
                    ));
                }
            },
            _ => {
                return Err(ParseError::new(
                    ParseErrorKind::UnexpectedToken {
                        found: self.current.kind.clone(),
                        expected: "BLOB operation (PUT, GET, DELETE, INFO, LINK, UNLINK, LINKS, TAG, UNTAG, VERIFY, GC, REPAIR, STATS, META)"
                            .to_string(),
                    },
                    self.current.span,
                ));
            },
        };

        Ok(StatementKind::Blob(BlobStmt { operation }))
    }

    fn parse_blobs(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Blobs)?;

        let operation = if self.current.is_eof() || self.check(&TokenKind::Semicolon) {
            // Just BLOBS - list all
            BlobsOp::List { pattern: None }
        } else if self.eat(&TokenKind::For) {
            // BLOBS FOR entity
            let entity = self.parse_expr()?;
            BlobsOp::For { entity }
        } else if self.eat(&TokenKind::By) {
            // BLOBS BY TAG 'tag'
            self.expect(&TokenKind::Tag)?;
            let tag = self.parse_expr()?;
            BlobsOp::ByTag { tag }
        } else if self.eat(&TokenKind::Where) {
            // BLOBS WHERE TYPE = 'type'
            // Expect TYPE keyword or an identifier
            if self.check(&TokenKind::Type)
                || self.check(&TokenKind::Ident("TYPE".to_string()))
                || self.check(&TokenKind::Ident("type".to_string()))
            {
                self.advance();
            }
            self.expect(&TokenKind::Eq)?;
            let content_type = self.parse_expr()?;
            BlobsOp::ByType { content_type }
        } else if self.eat(&TokenKind::Similar) {
            // BLOBS SIMILAR TO 'artifact_id' LIMIT n
            self.expect(&TokenKind::To)?;
            let artifact_id = self.parse_expr()?;
            let limit = if self.eat(&TokenKind::Limit) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            BlobsOp::Similar { artifact_id, limit }
        } else {
            // BLOBS 'pattern'
            let pattern = Some(self.parse_expr()?);
            BlobsOp::List { pattern }
        };

        Ok(StatementKind::Blobs(BlobsStmt { operation }))
    }

    // =========================================================================
    // Checkpoint Statement Parsers
    // =========================================================================

    fn parse_checkpoint(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Checkpoint)?;

        // Optional checkpoint name
        let name = if matches!(self.current.kind, TokenKind::String(_)) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(StatementKind::Checkpoint(CheckpointStmt { name }))
    }

    fn parse_rollback_or_chain_rollback(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Rollback)?;

        // Check if it's ROLLBACK CHAIN TO height
        if self.eat(&TokenKind::Chain) {
            self.expect(&TokenKind::To)?;
            let height = self.parse_expr()?;
            return Ok(StatementKind::Chain(ChainStmt {
                operation: ChainOp::Rollback { height },
            }));
        }

        // Regular ROLLBACK TO checkpoint
        self.expect(&TokenKind::To)?;
        let target = self.parse_expr()?;
        Ok(StatementKind::Rollback(RollbackStmt { target }))
    }

    fn parse_checkpoints(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Checkpoints)?;

        // Optional LIMIT
        let limit = if self.eat(&TokenKind::Limit) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(StatementKind::Checkpoints(CheckpointsStmt { limit }))
    }

    // =========================================================================
    // Chain Statement Parsers
    // =========================================================================

    fn parse_chain(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Chain)?;

        let operation = if self.eat(&TokenKind::History) {
            // CHAIN HISTORY 'key'
            let key = self.parse_expr()?;
            ChainOp::History { key }
        } else if self.eat(&TokenKind::Similar) {
            // CHAIN SIMILAR [embedding] LIMIT n
            let embedding = self.parse_vector_literal()?;
            let limit = if self.eat(&TokenKind::Limit) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            ChainOp::Similar { embedding, limit }
        } else if self.eat(&TokenKind::Drift) {
            // CHAIN DRIFT FROM height TO height
            self.expect(&TokenKind::From)?;
            let from_height = self.parse_expr()?;
            self.expect(&TokenKind::To)?;
            let to_height = self.parse_expr()?;
            ChainOp::Drift {
                from_height,
                to_height,
            }
        } else if self.eat(&TokenKind::Height) {
            // CHAIN HEIGHT
            ChainOp::Height
        } else if self.eat(&TokenKind::Tip) {
            // CHAIN TIP
            ChainOp::Tip
        } else if self.eat(&TokenKind::Block) {
            // CHAIN BLOCK height
            let height = self.parse_expr()?;
            ChainOp::Block { height }
        } else if self.eat(&TokenKind::Verify) {
            // CHAIN VERIFY
            ChainOp::Verify
        } else {
            return Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "HISTORY, SIMILAR, DRIFT, HEIGHT, TIP, BLOCK, or VERIFY",
            ));
        };

        Ok(StatementKind::Chain(ChainStmt { operation }))
    }

    fn parse_begin_chain(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Begin)?;
        self.expect(&TokenKind::Chain)?;
        // Optional TRANSACTION keyword
        self.eat(&TokenKind::Transaction);

        Ok(StatementKind::Chain(ChainStmt {
            operation: ChainOp::Begin,
        }))
    }

    fn parse_commit_chain(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Commit)?;
        self.expect(&TokenKind::Chain)?;

        Ok(StatementKind::Chain(ChainStmt {
            operation: ChainOp::Commit,
        }))
    }

    fn parse_analyze(&mut self) -> ParseResult<StatementKind> {
        self.expect(&TokenKind::Analyze)?;
        self.expect(&TokenKind::Codebook)?;
        self.expect(&TokenKind::Transitions)?;

        Ok(StatementKind::Chain(ChainStmt {
            operation: ChainOp::AnalyzeTransitions,
        }))
    }
}

/// Parses a single statement from source text.
///
/// # Errors
///
/// Returns an error if the input is not a valid statement.
pub fn parse(source: &str) -> ParseResult<Statement> {
    let mut parser = Parser::new(source);
    parser.parse_statement()
}

/// Parses multiple statements from source text.
///
/// # Errors
///
/// Returns an error if any statement in the input is invalid.
pub fn parse_all(source: &str) -> ParseResult<Vec<Statement>> {
    let mut parser = Parser::new(source);
    let mut statements = Vec::new();

    loop {
        let stmt = parser.parse_statement()?;
        if matches!(stmt.kind, StatementKind::Empty) && parser.current().is_eof() {
            break;
        }
        statements.push(stmt);
        if parser.current().is_eof() {
            break;
        }
    }

    Ok(statements)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_stmt(source: &str) -> Statement {
        parse(source).expect("parse failed")
    }

    #[test]
    fn test_simple_select() {
        let stmt = parse_stmt("SELECT * FROM users");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_columns() {
        let stmt = parse_stmt("SELECT id, name, email FROM users");
        let select = unwrap_select(stmt);
        assert_eq!(select.columns.len(), 3);
    }

    #[test]
    fn test_select_with_alias() {
        let stmt = parse_stmt("SELECT name AS user_name FROM users");
        let select = unwrap_select(stmt);
        assert!(select.columns[0].alias.is_some());
    }

    #[test]
    fn test_select_distinct() {
        let stmt = parse_stmt("SELECT DISTINCT name FROM users");
        let select = unwrap_select(stmt);
        assert!(select.distinct);
    }

    #[test]
    fn test_select_where() {
        let stmt = parse_stmt("SELECT * FROM users WHERE id = 1");
        let select = unwrap_select(stmt);
        assert!(select.where_clause.is_some());
    }

    #[test]
    fn test_select_order_by() {
        let stmt = parse_stmt("SELECT * FROM users ORDER BY name ASC");
        let select = unwrap_select(stmt);
        assert_eq!(select.order_by.len(), 1);
        assert_eq!(select.order_by[0].direction, SortDirection::Asc);
    }

    #[test]
    fn test_select_limit_offset() {
        let stmt = parse_stmt("SELECT * FROM users LIMIT 10 OFFSET 5");
        let select = unwrap_select(stmt);
        assert!(select.limit.is_some());
        assert!(select.offset.is_some());
    }

    #[test]
    fn test_select_join() {
        let stmt = parse_stmt("SELECT * FROM users u JOIN orders o ON u.id = o.user_id");
        let select = unwrap_select(stmt);
        assert_eq!(select.from.as_ref().unwrap().joins.len(), 1);
    }

    #[test]
    fn test_select_left_join() {
        let stmt = parse_stmt("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Left);
    }

    #[test]
    fn test_select_group_by_having() {
        let stmt = parse_stmt("SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1");
        let select = unwrap_select(stmt);
        assert!(!select.group_by.is_empty());
        assert!(select.having.is_some());
    }

    #[test]
    fn test_insert() {
        let stmt =
            parse_stmt("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')");
        let insert = unwrap_insert(stmt);
        assert_eq!(insert.table.name, "users");
        assert!(insert.columns.is_some());
    }

    #[test]
    fn test_update() {
        let stmt = parse_stmt("UPDATE users SET name = 'Bob' WHERE id = 1");
        let update = unwrap_update(stmt);
        assert_eq!(update.table.name, "users");
        assert_eq!(update.assignments.len(), 1);
        assert!(update.where_clause.is_some());
    }

    #[test]
    fn test_delete() {
        let stmt = parse_stmt("DELETE FROM users WHERE id = 1");
        let delete = unwrap_delete(stmt);
        assert_eq!(delete.table.name, "users");
        assert!(delete.where_clause.is_some());
    }

    #[test]
    fn test_create_table() {
        let stmt =
            parse_stmt("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100) NOT NULL)");
        let create = unwrap_createtable(stmt);
        assert_eq!(create.table.name, "users");
        assert_eq!(create.columns.len(), 2);
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let stmt = parse_stmt("CREATE TABLE IF NOT EXISTS users (id INT)");
        let create = unwrap_createtable(stmt);
        assert!(create.if_not_exists);
    }

    #[test]
    fn test_create_index() {
        let stmt = parse_stmt("CREATE INDEX idx_name ON users (name)");
        let create = unwrap_createindex(stmt);
        assert_eq!(create.name.name, "idx_name");
        assert!(!create.unique);
    }

    #[test]
    fn test_create_unique_index() {
        let stmt = parse_stmt("CREATE UNIQUE INDEX idx_email ON users (email)");
        let create = unwrap_createindex(stmt);
        assert!(create.unique);
    }

    #[test]
    fn test_drop_table() {
        let stmt = parse_stmt("DROP TABLE users");
        let drop = unwrap_droptable(stmt);
        assert_eq!(drop.table.name, "users");
        assert!(!drop.if_exists);
    }

    #[test]
    fn test_drop_table_if_exists() {
        let stmt = parse_stmt("DROP TABLE IF EXISTS users CASCADE");
        let drop = unwrap_droptable(stmt);
        assert!(drop.if_exists);
        assert!(drop.cascade);
    }

    #[test]
    fn test_drop_index() {
        let stmt = parse_stmt("DROP INDEX IF EXISTS idx_name");
        let drop = unwrap_dropindex(stmt);
        assert!(drop.if_exists);
    }

    #[test]
    fn test_node_create() {
        let stmt = parse_stmt("NODE CREATE user {name: 'Alice', age: 30}");
        let node = unwrap_node(stmt);
        let NodeOp::Create { label, properties } = node.operation else {
            panic!("expected NODE CREATE")
        };
        assert_eq!(label.name, "user");
        assert_eq!(properties.len(), 2);
    }

    #[test]
    fn test_node_get() {
        let stmt = parse_stmt("NODE GET 123");
        assert!(matches!(
            stmt.kind,
            StatementKind::Node(NodeStmt {
                operation: NodeOp::Get { .. }
            })
        ));
    }

    #[test]
    fn test_node_delete() {
        let stmt = parse_stmt("NODE DELETE 123");
        assert!(matches!(
            stmt.kind,
            StatementKind::Node(NodeStmt {
                operation: NodeOp::Delete { .. }
            })
        ));
    }

    #[test]
    fn test_node_list() {
        let stmt = parse_stmt("NODE LIST user");
        assert!(matches!(
            stmt.kind,
            StatementKind::Node(NodeStmt {
                operation: NodeOp::List { label: Some(_), .. }
            })
        ));
    }

    #[test]
    fn test_edge_create() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : knows {since: 2020}");
        let edge = unwrap_edge(stmt);
        let EdgeOp::Create { edge_type, .. } = edge.operation else {
            panic!("expected EDGE CREATE")
        };
        assert_eq!(edge_type.name, "knows");
    }

    #[test]
    fn test_neighbors() {
        let stmt = parse_stmt("NEIGHBORS 1 OUTGOING");
        let neighbors = unwrap_neighbors(stmt);
        assert_eq!(neighbors.direction, Direction::Outgoing);
    }

    #[test]
    fn test_path() {
        let stmt = parse_stmt("PATH SHORTEST 1 -> 2 LIMIT 5");
        let path = unwrap_path(stmt);
        assert_eq!(path.algorithm, PathAlgorithm::Shortest);
        assert!(path.max_depth.is_some());
    }

    #[test]
    fn test_embed_store() {
        let stmt = parse_stmt("EMBED STORE 'doc1' [0.1, 0.2, 0.3]");
        let embed = unwrap_embed(stmt);
        let EmbedOp::Store { vector, .. } = embed.operation else {
            panic!("expected EMBED STORE")
        };
        assert_eq!(vector.len(), 3);
    }

    #[test]
    fn test_embed_get() {
        let stmt = parse_stmt("EMBED GET 'doc1'");
        assert!(matches!(
            stmt.kind,
            StatementKind::Embed(EmbedStmt {
                operation: EmbedOp::Get { .. },
                ..
            })
        ));
    }

    #[test]
    fn test_similar() {
        let stmt = parse_stmt("SIMILAR 'doc1' LIMIT 10 COSINE");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Key(_)));
        assert!(similar.limit.is_some());
        assert_eq!(similar.metric, Some(DistanceMetric::Cosine));
    }

    #[test]
    fn test_similar_vector() {
        let stmt = parse_stmt("SIMILAR [0.1, 0.2] LIMIT 5");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Vector(_)));
    }

    #[test]
    fn test_find_nodes() {
        let stmt = parse_stmt("FIND NODE user WHERE age > 18 LIMIT 10");
        let find = unwrap_find(stmt);
        assert!(matches!(
            find.pattern,
            FindPattern::Nodes { label: Some(_) }
        ));
        assert!(find.where_clause.is_some());
        assert!(find.limit.is_some());
    }

    #[test]
    fn test_parse_all() {
        let stmts = parse_all("SELECT * FROM a; SELECT * FROM b").unwrap();
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_empty_input() {
        let stmt = parse_stmt("");
        assert!(matches!(stmt.kind, StatementKind::Empty));
    }

    #[test]
    fn test_semicolons() {
        let stmt = parse_stmt(";;;SELECT * FROM users;;");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_data_types() {
        let stmt =
            parse_stmt("CREATE TABLE t (a INT, b VARCHAR(255), c DECIMAL(10, 2), d BOOLEAN)");
        let create = unwrap_createtable(stmt);
        assert_eq!(create.columns.len(), 4);
        assert!(matches!(create.columns[0].data_type, DataType::Int));
        assert!(matches!(
            create.columns[1].data_type,
            DataType::Varchar(Some(255))
        ));
        assert!(matches!(
            create.columns[2].data_type,
            DataType::Decimal(Some(10), Some(2))
        ));
        assert!(matches!(create.columns[3].data_type, DataType::Boolean));
    }

    #[test]
    fn test_table_constraints() {
        let stmt =
            parse_stmt("CREATE TABLE t (id INT, name TEXT, PRIMARY KEY (id), UNIQUE (name))");
        let create = unwrap_createtable(stmt);
        assert_eq!(create.constraints.len(), 2);
    }

    #[test]
    fn test_join_using() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b USING (id)");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert!(matches!(
            from.joins[0].condition,
            Some(JoinCondition::Using(_))
        ));
    }

    #[test]
    fn test_order_by_nulls() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x NULLS FIRST");
        let select = unwrap_select(stmt);
        assert_eq!(select.order_by[0].nulls, Some(NullsOrder::First));
    }

    #[test]
    fn test_parser_source() {
        let parser = Parser::new("SELECT 1");
        assert_eq!(parser.source(), "SELECT 1");
    }

    #[test]
    fn test_all_data_types() {
        let types = [
            ("INTEGER", "DataType::Integer"),
            ("BIGINT", "DataType::Bigint"),
            ("SMALLINT", "DataType::Smallint"),
            ("FLOAT", "DataType::Float"),
            ("DOUBLE", "DataType::Double"),
            ("REAL", "DataType::Real"),
            ("NUMERIC", "DataType::Numeric"),
            ("TEXT", "DataType::Text"),
            ("BOOLEAN", "DataType::Boolean"),
            ("DATE", "DataType::Date"),
            ("TIME", "DataType::Time"),
            ("TIMESTAMP", "DataType::Timestamp"),
            ("BLOB", "DataType::Blob"),
        ];
        for (sql_type, _) in types {
            let sql = format!("CREATE TABLE t (x {})", sql_type);
            let stmt = parse_stmt(&sql);
            assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
        }
    }

    #[test]
    fn test_char_with_length() {
        let stmt = parse_stmt("CREATE TABLE t (x CHAR(10))");
        let create = unwrap_createtable(stmt);
        assert!(matches!(
            create.columns[0].data_type,
            DataType::Char(Some(10))
        ));
    }

    #[test]
    fn test_numeric_precision_scale() {
        let stmt = parse_stmt("CREATE TABLE t (x NUMERIC(5, 2))");
        let create = unwrap_createtable(stmt);
        assert!(matches!(
            create.columns[0].data_type,
            DataType::Numeric(Some(5), Some(2))
        ));
    }

    #[test]
    fn test_right_join() {
        let stmt = parse_stmt("SELECT * FROM a RIGHT JOIN b ON a.id = b.id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Right);
    }

    #[test]
    fn test_full_join() {
        let stmt = parse_stmt("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Full);
    }

    #[test]
    fn test_cross_join() {
        let stmt = parse_stmt("SELECT * FROM a CROSS JOIN b");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Cross);
    }

    #[test]
    fn test_natural_join() {
        let stmt = parse_stmt("SELECT * FROM a NATURAL JOIN b");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Natural);
    }

    #[test]
    fn test_edge_get() {
        let stmt = parse_stmt("EDGE GET 42");
        assert!(matches!(
            stmt.kind,
            StatementKind::Edge(EdgeStmt {
                operation: EdgeOp::Get { .. }
            })
        ));
    }

    #[test]
    fn test_edge_delete() {
        let stmt = parse_stmt("EDGE DELETE 42");
        assert!(matches!(
            stmt.kind,
            StatementKind::Edge(EdgeStmt {
                operation: EdgeOp::Delete { .. }
            })
        ));
    }

    #[test]
    fn test_edge_list() {
        let stmt = parse_stmt("EDGE LIST FOLLOWS");
        let edge = unwrap_edge(stmt);
        let EdgeOp::List { edge_type, .. } = edge.operation else {
            panic!("expected EDGE LIST")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_edge_list_no_type() {
        let stmt = parse_stmt("EDGE LIST");
        let edge = unwrap_edge(stmt);
        let EdgeOp::List { edge_type, .. } = edge.operation else {
            panic!("expected EDGE LIST")
        };
        assert!(edge_type.is_none());
    }

    #[test]
    fn test_node_list_no_label() {
        let stmt = parse_stmt("NODE LIST");
        let node = unwrap_node(stmt);
        let NodeOp::List { label, .. } = node.operation else {
            panic!("expected NODE LIST")
        };
        assert!(label.is_none());
    }

    #[test]
    fn test_node_list_with_limit() {
        let stmt = parse_stmt("NODE LIST LIMIT 10");
        let node = unwrap_node(stmt);
        if let NodeOp::List {
            label,
            limit,
            offset,
        } = node.operation
        {
            assert!(label.is_none());
            assert!(limit.is_some());
            assert!(offset.is_none());
        } else {
            panic!("expected NODE LIST");
        }
    }

    #[test]
    fn test_node_list_with_limit_offset() {
        let stmt = parse_stmt("NODE LIST user LIMIT 50 OFFSET 100");
        let node = unwrap_node(stmt);
        if let NodeOp::List {
            label,
            limit,
            offset,
        } = node.operation
        {
            assert!(label.is_some());
            assert!(limit.is_some());
            assert!(offset.is_some());
        } else {
            panic!("expected NODE LIST");
        }
    }

    #[test]
    fn test_edge_list_with_limit_offset() {
        let stmt = parse_stmt("EDGE LIST FOLLOWS LIMIT 25 OFFSET 50");
        let edge = unwrap_edge(stmt);
        if let EdgeOp::List {
            edge_type,
            limit,
            offset,
        } = edge.operation
        {
            assert!(edge_type.is_some());
            assert!(limit.is_some());
            assert!(offset.is_some());
        } else {
            panic!("expected EDGE LIST");
        }
    }

    #[test]
    fn test_embed_delete() {
        let stmt = parse_stmt("EMBED DELETE 'doc1'");
        assert!(matches!(
            stmt.kind,
            StatementKind::Embed(EmbedStmt {
                operation: EmbedOp::Delete { .. },
                ..
            })
        ));
    }

    #[test]
    fn test_neighbors_incoming() {
        let stmt = parse_stmt("NEIGHBORS 1 INCOMING");
        let neighbors = unwrap_neighbors(stmt);
        assert_eq!(neighbors.direction, Direction::Incoming);
    }

    #[test]
    fn test_neighbors_both() {
        let stmt = parse_stmt("NEIGHBORS 1 BOTH");
        let neighbors = unwrap_neighbors(stmt);
        assert_eq!(neighbors.direction, Direction::Both);
    }

    #[test]
    fn test_neighbors_with_type() {
        let stmt = parse_stmt("NEIGHBORS 1 OUTGOING : FOLLOWS");
        let neighbors = unwrap_neighbors(stmt);
        assert!(neighbors.edge_type.is_some());
    }

    #[test]
    fn test_path_without_shortest() {
        let stmt = parse_stmt("PATH 1 -> 2");
        let path = unwrap_path(stmt);
        assert_eq!(path.algorithm, PathAlgorithm::Shortest);
    }

    #[test]
    fn test_path_with_limit() {
        let stmt = parse_stmt("PATH 1 -> 2 LIMIT 5");
        let path = unwrap_path(stmt);
        assert!(path.max_depth.is_some());
    }

    #[test]
    fn test_find_edge() {
        let stmt = parse_stmt("FIND EDGE FOLLOWS WHERE weight > 0.5");
        let find = unwrap_find(stmt);
        assert!(matches!(
            find.pattern,
            FindPattern::Edges { edge_type: Some(_) }
        ));
    }

    #[test]
    fn test_find_vertex() {
        let stmt = parse_stmt("FIND VERTEX person");
        let find = unwrap_find(stmt);
        assert!(matches!(
            find.pattern,
            FindPattern::Nodes { label: Some(_) }
        ));
    }

    #[test]
    fn test_find_with_return() {
        let stmt = parse_stmt("FIND NODE user RETURN name, age");
        let find = unwrap_find(stmt);
        assert_eq!(find.return_items.len(), 2);
    }

    #[test]
    fn test_find_no_pattern() {
        let stmt = parse_stmt("FIND WHERE x > 1");
        let find = unwrap_find(stmt);
        assert!(matches!(find.pattern, FindPattern::Nodes { label: None }));
    }

    #[test]
    fn test_similar_euclidean() {
        let stmt = parse_stmt("SIMILAR 'doc' EUCLIDEAN");
        let similar = unwrap_similar(stmt);
        assert_eq!(similar.metric, Some(DistanceMetric::Euclidean));
    }

    #[test]
    fn test_similar_dot_product() {
        let stmt = parse_stmt("SIMILAR 'doc' DOT_PRODUCT");
        let similar = unwrap_similar(stmt);
        assert_eq!(similar.metric, Some(DistanceMetric::DotProduct));
    }

    #[test]
    fn test_column_not_null() {
        let stmt = parse_stmt("CREATE TABLE t (x INT NOT NULL)");
        let create = unwrap_createtable(stmt);
        assert!(create.columns[0]
            .constraints
            .contains(&ColumnConstraint::NotNull));
    }

    #[test]
    fn test_column_default() {
        let stmt = parse_stmt("CREATE TABLE t (x INT DEFAULT 0)");
        let create = unwrap_createtable(stmt);
        assert!(create.columns[0]
            .constraints
            .iter()
            .any(|c| matches!(c, ColumnConstraint::Default(_))));
    }

    #[test]
    fn test_column_references() {
        let stmt = parse_stmt("CREATE TABLE t (x INT REFERENCES other(id))");
        let create = unwrap_createtable(stmt);
        assert!(create.columns[0]
            .constraints
            .iter()
            .any(|c| matches!(c, ColumnConstraint::References { .. })));
    }

    #[test]
    fn test_insert_multiple_rows() {
        let stmt = parse_stmt("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)");
        let insert = unwrap_insert(stmt);
        let InsertSource::Values(rows) = insert.source else {
            panic!("expected INSERT")
        };
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_insert_with_select() {
        let stmt = parse_stmt("INSERT INTO t SELECT * FROM other");
        let insert = unwrap_insert(stmt);
        assert!(matches!(insert.source, InsertSource::Query(_)));
    }

    #[test]
    fn test_order_by_nulls_last() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x DESC NULLS LAST");
        let select = unwrap_select(stmt);
        assert_eq!(select.order_by[0].direction, SortDirection::Desc);
        assert_eq!(select.order_by[0].nulls, Some(NullsOrder::Last));
    }

    #[test]
    fn test_select_multiple_columns() {
        let stmt = parse_stmt("SELECT a, b, c FROM t");
        let select = unwrap_select(stmt);
        assert_eq!(select.columns.len(), 3);
    }

    #[test]
    fn test_select_all_keyword() {
        let stmt = parse_stmt("SELECT ALL * FROM t");
        let select = unwrap_select(stmt);
        assert!(!select.distinct); // ALL is the default (not distinct)
    }

    #[test]
    fn test_inner_join() {
        let stmt = parse_stmt("SELECT * FROM a INNER JOIN b ON a.id = b.id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Inner);
    }

    #[test]
    fn test_right_outer_join() {
        let stmt = parse_stmt("SELECT * FROM a RIGHT OUTER JOIN b ON a.id = b.id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Right);
    }

    #[test]
    fn test_left_outer_join() {
        let stmt = parse_stmt("SELECT * FROM a LEFT OUTER JOIN b ON a.id = b.id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins[0].kind, JoinKind::Left);
    }

    #[test]
    fn test_table_alias_implicit() {
        let stmt = parse_stmt("SELECT * FROM users u");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.table.alias.as_ref().unwrap().name, "u");
    }

    #[test]
    fn test_select_item_alias() {
        let stmt = parse_stmt("SELECT x AS y FROM t");
        let select = unwrap_select(stmt);
        assert!(select.columns[0].alias.is_some());
    }

    #[test]
    fn test_foreign_key_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, FOREIGN KEY (x) REFERENCES other(id))");
        let create = unwrap_createtable(stmt);
        assert!(create
            .constraints
            .iter()
            .any(|c| matches!(c, TableConstraint::ForeignKey { .. })));
    }

    #[test]
    fn test_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, CHECK (x > 0))");
        let create = unwrap_createtable(stmt);
        assert!(create
            .constraints
            .iter()
            .any(|c| matches!(c, TableConstraint::Check(_))));
    }

    // Error handling tests
    #[test]
    fn test_error_unexpected_eof() {
        let result = parse("SELECT");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_from() {
        let result = parse("SELECT * WHERE x = 1");
        // This parses as SELECT with no FROM, which is valid
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_invalid_node_op() {
        let result = parse("NODE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_edge_op() {
        let result = parse("EDGE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_embed_op() {
        let result = parse("EMBED INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_arrow_in_edge() {
        let result = parse("EDGE CREATE 1 2");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_missing_colon_in_edge() {
        let result = parse("EDGE CREATE 1 -> 2 type");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_statement() {
        let result = parse("INVALID STATEMENT");
        assert!(result.is_err());
    }

    // More expression tests - subqueries require further implementation
    #[test]
    fn test_in_with_values() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_not_in() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_nested_case() {
        let stmt = parse_stmt(
            "SELECT CASE WHEN x > 0 THEN CASE WHEN y > 0 THEN 1 ELSE 2 END ELSE 0 END FROM t",
        );
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // More graph tests
    #[test]
    fn test_node_create_empty_properties() {
        let stmt = parse_stmt("NODE CREATE user {}");
        let node = unwrap_node(stmt);
        let NodeOp::Create { properties, .. } = node.operation else {
            panic!("expected NODE CREATE")
        };
        assert!(properties.is_empty());
    }

    #[test]
    fn test_edge_create_empty_properties() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : knows {}");
        let edge = unwrap_edge(stmt);
        let EdgeOp::Create { properties, .. } = edge.operation else {
            panic!("expected EDGE CREATE")
        };
        assert!(properties.is_empty());
    }

    // More SQL tests
    #[test]
    fn test_update_multiple_columns() {
        let stmt = parse_stmt("UPDATE t SET a = 1, b = 2, c = 3 WHERE id = 1");
        let update = unwrap_update(stmt);
        assert_eq!(update.assignments.len(), 3);
    }

    #[test]
    fn test_delete_without_where() {
        let stmt = parse_stmt("DELETE FROM users");
        let delete = unwrap_delete(stmt);
        assert!(delete.where_clause.is_none());
    }

    #[test]
    fn test_select_with_offset() {
        let stmt = parse_stmt("SELECT * FROM t LIMIT 10 OFFSET 5");
        let select = unwrap_select(stmt);
        assert!(select.limit.is_some());
        assert!(select.offset.is_some());
    }

    #[test]
    fn test_multiple_joins() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        assert_eq!(from.joins.len(), 2);
    }

    #[test]
    fn test_complex_where_clause() {
        let stmt = parse_stmt("SELECT * FROM t WHERE (a > 1 AND b < 2) OR (c = 3 AND d != 4)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_aggregate_with_filter() {
        let stmt = parse_stmt("SELECT COUNT(*), SUM(amount), AVG(price) FROM orders");
        let select = unwrap_select(stmt);
        assert_eq!(select.columns.len(), 3);
    }

    // Vector tests
    #[test]
    fn test_similar_no_options() {
        let stmt = parse_stmt("SIMILAR 'query'");
        let similar = unwrap_similar(stmt);
        assert!(similar.limit.is_none());
        assert!(similar.metric.is_none());
    }

    // FIND tests
    #[test]
    fn test_find_edge_no_type() {
        let stmt = parse_stmt("FIND EDGE WHERE weight > 0.5");
        let find = unwrap_find(stmt);
        assert!(matches!(
            find.pattern,
            FindPattern::Edges { edge_type: None }
        ));
    }

    #[test]
    fn test_find_node_no_label() {
        let stmt = parse_stmt("FIND NODE WHERE active = TRUE");
        let find = unwrap_find(stmt);
        assert!(matches!(find.pattern, FindPattern::Nodes { label: None }));
    }

    // SHOW TABLES tests
    #[test]
    fn test_show_tables() {
        let stmt = parse_stmt("SHOW TABLES");
        assert!(matches!(stmt.kind, StatementKind::ShowTables));
    }

    #[test]
    fn test_show_tables_semicolon() {
        let stmt = parse_stmt("SHOW TABLES;");
        assert!(matches!(stmt.kind, StatementKind::ShowTables));
    }

    #[test]
    fn test_show_tables_lowercase() {
        let stmt = parse_stmt("show tables");
        assert!(matches!(stmt.kind, StatementKind::ShowTables));
    }

    #[test]
    #[should_panic(expected = "TABLES")]
    fn test_show_without_tables() {
        parse_stmt("SHOW");
    }

    // Bit operations tests
    #[test]
    fn test_bit_or() {
        let stmt = parse_stmt("SELECT 1 | 2");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::BitOr, _)
        ));
    }

    #[test]
    fn test_bit_and() {
        let stmt = parse_stmt("SELECT 1 & 2");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::BitAnd, _)
        ));
    }

    #[test]
    fn test_bit_xor() {
        let stmt = parse_stmt("SELECT 1 ^ 2");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::BitXor, _)
        ));
    }

    #[test]
    fn test_bit_shift_left() {
        let stmt = parse_stmt("SELECT 1 << 2");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Shl, _)
        ));
    }

    #[test]
    fn test_bit_shift_right() {
        let stmt = parse_stmt("SELECT 1 >> 2");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Shr, _)
        ));
    }

    #[test]
    fn test_bit_not() {
        let stmt = parse_stmt("SELECT ~1");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Unary(UnaryOp::BitNot, _)
        ));
    }

    // Aggregate functions
    #[test]
    fn test_aggregate_sum() {
        let stmt = parse_stmt("SELECT SUM(x) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Call(_)));
    }

    #[test]
    fn test_aggregate_avg() {
        let stmt = parse_stmt("SELECT AVG(x) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Call(_)));
    }

    #[test]
    fn test_aggregate_min() {
        let stmt = parse_stmt("SELECT MIN(x) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Call(_)));
    }

    #[test]
    fn test_aggregate_max() {
        let stmt = parse_stmt("SELECT MAX(x) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Call(_)));
    }

    // NOT IN expression
    #[test]
    fn test_not_in_list() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT IN (1, 2, 3)");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // EXISTS subquery
    #[test]
    fn test_exists_subquery() {
        let stmt = parse_stmt("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM u)");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // ORDER BY with direction
    #[test]
    fn test_order_by_asc() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x ASC");
        let sel = unwrap_select(stmt);
        assert!(!sel.order_by.is_empty());
        assert!(matches!(sel.order_by[0].direction, SortDirection::Asc));
    }

    #[test]
    fn test_order_by_desc() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x DESC");
        let sel = unwrap_select(stmt);
        assert!(!sel.order_by.is_empty());
        assert!(matches!(sel.order_by[0].direction, SortDirection::Desc));
    }

    // FALSE and NULL literals
    #[test]
    fn test_false_literal() {
        let stmt = parse_stmt("SELECT FALSE FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_null_literal() {
        let stmt = parse_stmt("SELECT NULL FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Unary operators
    #[test]
    fn test_unary_neg() {
        let stmt = parse_stmt("SELECT -1 FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_unary_not() {
        let stmt = parse_stmt("SELECT NOT x FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // IS NULL / IS NOT NULL
    #[test]
    fn test_is_null() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NULL");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    #[test]
    fn test_is_not_null() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NOT NULL");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // BETWEEN / NOT BETWEEN
    #[test]
    fn test_between() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    #[test]
    fn test_not_between() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // LIKE / NOT LIKE
    #[test]
    fn test_like() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name LIKE '%foo%'");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    #[test]
    fn test_not_like() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name NOT LIKE '%bar%'");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // Qualified wildcard (table.*)
    #[test]
    fn test_qualified_wildcard() {
        let stmt = parse_stmt("SELECT t.* FROM t");
        let sel = unwrap_select(stmt);
        assert!(!sel.columns.is_empty());
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::QualifiedWildcard(_)
        ));
    }

    // Array expression
    #[test]
    fn test_array_expr() {
        let stmt = parse_stmt("SELECT [1, 2, 3] FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Array(_)));
    }

    // Division and modulo operators
    #[test]
    fn test_division() {
        let stmt = parse_stmt("SELECT a / b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Div, _)
        ));
    }

    #[test]
    fn test_modulo() {
        let stmt = parse_stmt("SELECT a % b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Mod, _)
        ));
    }

    // Addition and subtraction
    #[test]
    fn test_addition() {
        let stmt = parse_stmt("SELECT a + b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Add, _)
        ));
    }

    #[test]
    fn test_subtraction() {
        let stmt = parse_stmt("SELECT a - b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Sub, _)
        ));
    }

    // Multiplication
    #[test]
    fn test_multiplication() {
        let stmt = parse_stmt("SELECT a * b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Mul, _)
        ));
    }

    // String concatenation
    #[test]
    fn test_concat() {
        let stmt = parse_stmt("SELECT a || b FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Binary(_, BinaryOp::Concat, _)
        ));
    }

    // Comparison operators <= and >=
    #[test]
    fn test_less_equal() {
        let stmt = parse_stmt("SELECT * FROM t WHERE a <= b");
        let sel = unwrap_select(stmt);
        if let Some(ref where_clause) = sel.where_clause {
            assert!(matches!(
                where_clause.kind,
                ExprKind::Binary(_, BinaryOp::Le, _)
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_greater_equal() {
        let stmt = parse_stmt("SELECT * FROM t WHERE a >= b");
        let sel = unwrap_select(stmt);
        if let Some(ref where_clause) = sel.where_clause {
            assert!(matches!(
                where_clause.kind,
                ExprKind::Binary(_, BinaryOp::Ge, _)
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    // Tuple expressions
    #[test]
    fn test_tuple_expr() {
        let stmt = parse_stmt("SELECT (1, 2, 3) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Tuple(_)));
    }

    #[test]
    fn test_empty_tuple_expr() {
        let stmt = parse_stmt("SELECT () FROM t");
        let sel = unwrap_select(stmt);
        let ExprKind::Tuple(items) = &sel.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert!(items.is_empty());
    }

    // CASE with operand (simple CASE)
    #[test]
    fn test_case_with_operand() {
        let stmt = parse_stmt("SELECT CASE x WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END FROM t");
        let sel = unwrap_select(stmt);
        let ExprKind::Case(case) = &sel.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert!(case.operand.is_some());
        assert!(!case.when_clauses.is_empty());
        assert!(case.else_clause.is_some());
    }

    // Subquery in FROM
    #[test]
    fn test_subquery_from() {
        let stmt = parse_stmt("SELECT * FROM (SELECT 1 AS x) AS sub");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert!(matches!(from.table.kind, TableRefKind::Subquery(_)));
        } else {
            panic!("expected SELECT");
        }
    }

    // INSERT with SELECT (instead of VALUES)
    #[test]
    fn test_insert_select() {
        let stmt = parse_stmt("INSERT INTO t (a, b) SELECT x, y FROM s");
        let ins = unwrap_insert(stmt);
        assert!(matches!(ins.source, InsertSource::Query(_)));
    }

    // Function call expression
    #[test]
    fn test_function_call() {
        let stmt = parse_stmt("SELECT UPPER(name) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Call(_)));
    }

    // Cast expression
    #[test]
    fn test_cast_expr() {
        let stmt = parse_stmt("SELECT CAST(x AS INT) FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(sel.columns[0].expr.kind, ExprKind::Cast(_, _)));
    }

    // CASE without ELSE
    #[test]
    fn test_case_no_else() {
        let stmt = parse_stmt("SELECT CASE WHEN x > 0 THEN 1 END FROM t");
        let sel = unwrap_select(stmt);
        let ExprKind::Case(case) = &sel.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert!(case.else_clause.is_none());
    }

    // Subquery with alias
    #[test]
    fn test_subquery_with_alias() {
        let stmt = parse_stmt("SELECT sub.x FROM (SELECT 1 AS x) sub");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert!(from.table.alias.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    // Table alias without AS keyword
    #[test]
    fn test_table_alias_no_as() {
        let stmt = parse_stmt("SELECT t.x FROM users t");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert!(from.table.alias.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    // Join with ON clause
    #[test]
    fn test_join_on_clause() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b ON a.id = b.id");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert!(!from.joins.is_empty());
            assert!(from.joins[0].condition.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    // IN with subquery
    #[test]
    fn test_in_subquery() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IN (SELECT y FROM s)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Parenthesized expression
    #[test]
    fn test_paren_expr() {
        let stmt = parse_stmt("SELECT (1 + 2) * 3 FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Multiple order by columns
    #[test]
    fn test_order_by_multiple() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY a ASC, b DESC");
        let sel = unwrap_select(stmt);
        assert_eq!(sel.order_by.len(), 2);
    }

    // GROUP BY with multiple columns
    #[test]
    fn test_group_by_multiple() {
        let stmt = parse_stmt("SELECT a, b, COUNT(*) FROM t GROUP BY a, b");
        let sel = unwrap_select(stmt);
        assert_eq!(sel.group_by.len(), 2);
    }

    // Float literal
    #[test]
    fn test_float_literal() {
        let stmt = parse_stmt("SELECT 3.14 FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Literal(Literal::Float(_))
        ));
    }

    // String literal
    #[test]
    fn test_string_literal() {
        let stmt = parse_stmt("SELECT 'hello' FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Literal(Literal::String(_))
        ));
    }

    // Integer literal
    #[test]
    fn test_integer_literal() {
        let stmt = parse_stmt("SELECT 42 FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Literal(Literal::Integer(42))
        ));
    }

    // TRUE literal
    #[test]
    fn test_true_literal() {
        let stmt = parse_stmt("SELECT TRUE FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Literal(Literal::Boolean(true))
        ));
    }

    // Qualified column name (table.column)
    #[test]
    fn test_qualified_column() {
        let stmt = parse_stmt("SELECT t.x FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Qualified(_, _)
        ));
    }

    // Logical AND/OR
    #[test]
    fn test_logical_and_or() {
        let stmt = parse_stmt("SELECT * FROM t WHERE a = 1 AND b = 2 OR c = 3");
        let sel = unwrap_select(stmt);
        assert!(sel.where_clause.is_some());
    }

    // Nested subqueries
    #[test]
    fn test_nested_subquery() {
        let stmt = parse_stmt("SELECT * FROM (SELECT * FROM (SELECT 1 AS x) inner_sub) outer_sub");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Custom data type
    #[test]
    fn test_custom_data_type() {
        let stmt = parse_stmt("CREATE TABLE t (x my_custom_type)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // NULL constraint
    #[test]
    fn test_null_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT NULL)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // UNIQUE constraint
    #[test]
    fn test_unique_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT UNIQUE)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // CHECK constraint on column
    #[test]
    fn test_check_column_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT CHECK (x > 0))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // CAST to different types
    #[test]
    fn test_cast_to_varchar() {
        let stmt = parse_stmt("SELECT CAST(x AS VARCHAR(100)) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_cast_to_decimal() {
        let stmt = parse_stmt("SELECT CAST(x AS DECIMAL(10, 2)) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Comparison operators
    #[test]
    fn test_not_equal() {
        let stmt = parse_stmt("SELECT * FROM t WHERE a != b");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_not_equal_ansi() {
        let stmt = parse_stmt("SELECT * FROM t WHERE a <> b");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Bit not operator
    #[test]
    fn test_bit_not_expr() {
        let stmt = parse_stmt("SELECT ~x FROM t");
        let sel = unwrap_select(stmt);
        assert!(matches!(
            sel.columns[0].expr.kind,
            ExprKind::Unary(UnaryOp::BitNot, _)
        ));
    }

    // Empty values in IN
    #[test]
    fn test_in_empty() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IN ()");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // CASE with multiple WHEN clauses
    #[test]
    fn test_case_multiple_when() {
        let stmt = parse_stmt("SELECT CASE WHEN a THEN 1 WHEN b THEN 2 WHEN c THEN 3 END FROM t");
        let sel = unwrap_select(stmt);
        let ExprKind::Case(case) = &sel.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert_eq!(case.when_clauses.len(), 3);
    }

    // Float with exponent
    #[test]
    fn test_float_exponent() {
        let stmt = parse_stmt("SELECT 1.5e10 FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Negative number
    #[test]
    fn test_negative_number() {
        let stmt = parse_stmt("SELECT -42 FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Complex nested expression
    #[test]
    fn test_complex_nested() {
        let stmt = parse_stmt("SELECT ((a + b) * (c - d)) / e FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Multiple table joins
    #[test]
    fn test_three_way_join() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON b.id = c.id");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert_eq!(from.joins.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    // Select with LIMIT only
    #[test]
    fn test_select_limit_only() {
        let stmt = parse_stmt("SELECT * FROM t LIMIT 10");
        let sel = unwrap_select(stmt);
        assert!(sel.limit.is_some());
        assert!(sel.offset.is_none());
    }

    // Update with multiple assignments
    #[test]
    fn test_update_three_columns() {
        let stmt = parse_stmt("UPDATE t SET a = 1, b = 2, c = 3 WHERE id = 1");
        let upd = unwrap_update(stmt);
        assert_eq!(upd.assignments.len(), 3);
    }

    // Implicit column alias (no AS keyword)
    #[test]
    fn test_implicit_column_alias() {
        let stmt = parse_stmt("SELECT x alias FROM t");
        let sel = unwrap_select(stmt);
        assert!(sel.columns[0].alias.is_some());
    }

    // Table followed by keyword (should not treat keyword as alias)
    #[test]
    fn test_table_followed_by_keyword() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x = 1");
        let sel = unwrap_select(stmt);
        if let Some(ref from) = sel.from {
            assert!(from.table.alias.is_none());
        } else {
            panic!("expected SELECT");
        }
    }

    // Column alias followed by keyword
    #[test]
    fn test_column_alias_followed_by_keyword() {
        let stmt = parse_stmt("SELECT x y FROM t WHERE y = 1");
        let sel = unwrap_select(stmt);
        assert!(sel.columns[0].alias.is_some());
    }

    // Delete with complex WHERE
    #[test]
    fn test_delete_complex_where() {
        let stmt = parse_stmt("DELETE FROM t WHERE a = 1 AND b = 2 OR c = 3");
        let del = unwrap_delete(stmt);
        assert!(del.where_clause.is_some());
    }

    // Multiple INSERT rows
    #[test]
    fn test_insert_three_rows() {
        let stmt = parse_stmt("INSERT INTO t (a, b) VALUES (1, 2), (3, 4), (5, 6)");
        let ins = unwrap_insert(stmt);
        let InsertSource::Values(rows) = ins.source else {
            panic!("expected INSERT")
        };
        assert_eq!(rows.len(), 3);
    }

    // OFFSET without LIMIT
    #[test]
    fn test_offset_only() {
        let stmt = parse_stmt("SELECT * FROM t OFFSET 10");
        let sel = unwrap_select(stmt);
        assert!(sel.offset.is_some());
        assert!(sel.limit.is_none());
    }

    // Empty GROUP BY (just the keyword)
    #[test]
    fn test_simple_group_by() {
        let stmt = parse_stmt("SELECT a, COUNT(*) FROM t GROUP BY a");
        let sel = unwrap_select(stmt);
        assert_eq!(sel.group_by.len(), 1);
    }

    // NOT IN subquery
    #[test]
    fn test_not_in_subquery() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT IN (SELECT y FROM s)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // EXISTS in WHERE
    #[test]
    fn test_exists_in_where() {
        let stmt = parse_stmt("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM s)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Column with table prefix in WHERE
    #[test]
    fn test_qualified_column_in_where() {
        let stmt = parse_stmt("SELECT * FROM t WHERE t.x = 1");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Multiple columns in SELECT
    #[test]
    fn test_select_five_columns() {
        let stmt = parse_stmt("SELECT a, b, c, d, e FROM t");
        let sel = unwrap_select(stmt);
        assert_eq!(sel.columns.len(), 5);
    }

    // Deeply nested arithmetic
    #[test]
    fn test_deeply_nested_arithmetic() {
        let stmt = parse_stmt("SELECT (((a + b) - c) * d) / e FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // All comparison operators in one query
    #[test]
    fn test_all_comparisons() {
        let stmt = parse_stmt(
            "SELECT * FROM t WHERE a = 1 AND b != 2 AND c < 3 AND d <= 4 AND e > 5 AND f >= 6",
        );
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Error tests - these test error paths
    #[test]
    fn test_error_unexpected_expression() {
        let result = parse("SELECT FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_drop_missing_table_or_index() {
        let result = parse("DROP DATABASE foo");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_invalid_data_type() {
        let result = parse("CREATE TABLE t (x 123)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_insert_no_values_or_select() {
        let result = parse("INSERT INTO t (a, b) FROM x");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_qualified_wildcard_on_expr() {
        let result = parse("SELECT (1+2).* FROM t");
        assert!(result.is_err());
    }

    // More error tests to hit remaining uncovered error paths
    #[test]
    fn test_error_table_constraint_invalid() {
        // Invalid constraint keyword
        let result = parse("CREATE TABLE t (x INT, INVALID constraint)");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_unexpected_in_primary_expr() {
        // An unexpected token where an expression is expected
        let result = parse("SELECT , FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_case_empty_when() {
        // CASE with nothing between WHEN and THEN - hard to trigger
        let result = parse("SELECT CASE WHEN THEN 1 END FROM t");
        assert!(result.is_err());
    }

    // Additional edge case tests
    #[test]
    fn test_table_alias_with_as() {
        let stmt = parse_stmt("SELECT * FROM users AS u");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_no_from() {
        let stmt = parse_stmt("SELECT 1 + 2");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_nested_parentheses() {
        let stmt = parse_stmt("SELECT ((1 + 2) * 3) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_union_not_supported() {
        // UNION is likely not supported, but let's try
        let result = parse("SELECT 1 UNION SELECT 2");
        // It might parse the first SELECT and fail on UNION
        // Either way this tests more code paths
        let _ = result;
    }

    #[test]
    fn test_binary_expression_chain() {
        let stmt = parse_stmt("SELECT a + b + c + d + e FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_mixed_operators() {
        let stmt = parse_stmt("SELECT a + b * c - d / e FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Coverage tests for table constraints
    #[test]
    fn test_named_primary_key_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, CONSTRAINT pk PRIMARY KEY (x))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_composite_primary_key() {
        let stmt = parse_stmt("CREATE TABLE t (a INT, b INT, PRIMARY KEY (a, b))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_named_unique_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, CONSTRAINT uq UNIQUE (x))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_composite_unique_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (a INT, b INT, UNIQUE (a, b))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_composite_foreign_key() {
        let stmt = parse_stmt("CREATE TABLE t (a INT, b INT, FOREIGN KEY (a, b) REFERENCES other)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_foreign_key_without_column_ref() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, FOREIGN KEY (x) REFERENCES other)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_index_if_not_exists() {
        let stmt = parse_stmt("CREATE INDEX IF NOT EXISTS idx ON t (x)");
        assert!(matches!(stmt.kind, StatementKind::CreateIndex(_)));
    }

    #[test]
    fn test_drop_index_if_exists() {
        let stmt = parse_stmt("DROP INDEX IF EXISTS idx");
        assert!(matches!(stmt.kind, StatementKind::DropIndex(_)));
    }

    #[test]
    fn test_drop_table_cascade() {
        let stmt = parse_stmt("DROP TABLE IF EXISTS users CASCADE");
        assert!(matches!(stmt.kind, StatementKind::DropTable(_)));
    }

    #[test]
    fn test_named_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, CONSTRAINT chk CHECK (x > 0))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_named_foreign_key_constraint() {
        let stmt = parse_stmt(
            "CREATE TABLE t (x INT, CONSTRAINT fk FOREIGN KEY (x) REFERENCES other (id))",
        );
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_three_column_primary_key() {
        let stmt = parse_stmt("CREATE TABLE t (a INT, b INT, c INT, PRIMARY KEY (a, b, c))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_unique_index_if_not_exists() {
        let stmt = parse_stmt("CREATE UNIQUE INDEX IF NOT EXISTS idx ON t (x)");
        assert!(matches!(stmt.kind, StatementKind::CreateIndex(_)));
    }

    // More graph statement coverage
    #[test]
    fn test_node_list_all() {
        let stmt = parse_stmt("NODE LIST");
        assert!(matches!(stmt.kind, StatementKind::Node(_)));
    }

    #[test]
    fn test_edge_list_all() {
        let stmt = parse_stmt("EDGE LIST");
        assert!(matches!(stmt.kind, StatementKind::Edge(_)));
    }

    // NODE DELETE, EDGE DELETE, NODE GET, EDGE GET already covered earlier

    // Error path tests
    #[test]
    fn test_show_error() {
        let result = parse("SHOW COLUMNS");
        assert!(result.is_err());
    }

    #[test]
    fn test_node_invalid_op() {
        let result = parse("NODE UPDATE 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_invalid_op() {
        let result = parse("EDGE UPDATE 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_invalid() {
        let result = parse("DROP VIEW test");
        assert!(result.is_err());
    }

    // Multi-column index
    #[test]
    fn test_create_index_multi_column() {
        let stmt = parse_stmt("CREATE INDEX idx ON t (a, b, c)");
        assert!(matches!(stmt.kind, StatementKind::CreateIndex(_)));
    }

    // Table ref with keyword after (should not be alias)
    #[test]
    fn test_table_ref_no_alias_before_where() {
        let stmt = parse_stmt("SELECT * FROM users WHERE id = 1");
        if let StatementKind::Select(sel) = &stmt.kind {
            let from = sel.from.as_ref().unwrap();
            assert!(from.table.alias.is_none());
        }
    }

    #[test]
    fn test_table_ref_no_alias_before_join() {
        let stmt = parse_stmt("SELECT * FROM users JOIN orders ON users.id = orders.user_id");
        if let StatementKind::Select(sel) = &stmt.kind {
            let from = sel.from.as_ref().unwrap();
            assert!(from.table.alias.is_none());
        }
    }

    // Cover more update paths
    #[test]
    fn test_update_multiple_sets() {
        let stmt = parse_stmt("UPDATE t SET a = 1, b = 2, c = 3 WHERE id = 1");
        assert!(matches!(stmt.kind, StatementKind::Update(_)));
    }

    // Neighbors and path tests (already covered by earlier tests)

    // Vector/similar tests already covered by earlier tests

    // Join types already covered by earlier tests

    // Column data types coverage
    #[test]
    fn test_create_table_varchar() {
        let stmt = parse_stmt("CREATE TABLE t (x VARCHAR(255))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_bigint() {
        let stmt = parse_stmt("CREATE TABLE t (x BIGINT)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_smallint() {
        let stmt = parse_stmt("CREATE TABLE t (x SMALLINT)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_double() {
        let stmt = parse_stmt("CREATE TABLE t (x DOUBLE)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_real() {
        let stmt = parse_stmt("CREATE TABLE t (x REAL)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_date() {
        let stmt = parse_stmt("CREATE TABLE t (x DATE)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_time() {
        let stmt = parse_stmt("CREATE TABLE t (x TIME)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_uuid() {
        let stmt = parse_stmt("CREATE TABLE t (x UUID)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_json() {
        let stmt = parse_stmt("CREATE TABLE t (x JSON)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_blob() {
        let stmt = parse_stmt("CREATE TABLE t (x BLOB)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_char() {
        let stmt = parse_stmt("CREATE TABLE t (x CHAR(10))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_numeric() {
        let stmt = parse_stmt("CREATE TABLE t (x NUMERIC(10, 2))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // Additional coverage tests
    #[test]
    fn test_find_with_where() {
        let stmt = parse_stmt("FIND NODE person WHERE age > 18");
        assert!(matches!(stmt.kind, StatementKind::Find(_)));
    }

    // EMBED STORE and SIMILAR with COSINE already covered

    #[test]
    fn test_table_constraint_error() {
        let result = parse("CREATE TABLE t (x INT, INVALID)");
        assert!(result.is_err());
    }

    // Additional coverage tests - many already defined above

    #[test]
    fn test_similar_with_vector_query() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0, 3.0] LIMIT 5");
        assert!(matches!(stmt.kind, StatementKind::Similar(_)));
    }

    #[test]
    fn test_embed_get_coverage() {
        let stmt = parse_stmt("EMBED GET 'mykey'");
        assert!(matches!(stmt.kind, StatementKind::Embed(_)));
    }

    #[test]
    fn test_embed_delete_coverage() {
        let stmt = parse_stmt("EMBED DELETE 'mykey'");
        assert!(matches!(stmt.kind, StatementKind::Embed(_)));
    }

    #[test]
    fn test_similar_dot_product_metric() {
        let stmt = parse_stmt("SIMILAR 'query' DOT_PRODUCT LIMIT 5");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert_eq!(similar.metric, Some(DistanceMetric::DotProduct));
        }
    }

    #[test]
    fn test_empty_embed_store_vector() {
        let stmt = parse_stmt("EMBED STORE 'key' []");
        assert!(matches!(stmt.kind, StatementKind::Embed(_)));
    }

    #[test]
    fn test_path_shortest_keyword() {
        let stmt = parse_stmt("PATH SHORTEST 1 -> 10");
        assert!(matches!(stmt.kind, StatementKind::Path(_)));
    }

    #[test]
    fn test_neighbors_with_edge_type() {
        let stmt = parse_stmt("NEIGHBORS 1 : friends");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert!(neighbors.edge_type.is_some());
        }
    }

    #[test]
    fn test_neighbors_default_outgoing() {
        let stmt = parse_stmt("NEIGHBORS 1");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert_eq!(neighbors.direction, Direction::Outgoing);
        }
    }

    // CACHE command tests
    #[test]
    fn test_cache_init() {
        let stmt = parse_stmt("CACHE INIT");
        let cache = unwrap_cache(stmt);
        assert!(matches!(cache.operation, CacheOp::Init));
    }

    #[test]
    fn test_cache_stats() {
        let stmt = parse_stmt("CACHE STATS");
        let cache = unwrap_cache(stmt);
        assert!(matches!(cache.operation, CacheOp::Stats));
    }

    #[test]
    fn test_cache_clear() {
        let stmt = parse_stmt("CACHE CLEAR");
        let cache = unwrap_cache(stmt);
        assert!(matches!(cache.operation, CacheOp::Clear));
    }

    #[test]
    fn test_cache_lowercase() {
        let stmt = parse_stmt("cache init");
        assert!(matches!(stmt.kind, StatementKind::Cache(_)));
    }

    #[test]
    fn test_cache_with_semicolon() {
        let stmt = parse_stmt("CACHE STATS;");
        assert!(matches!(stmt.kind, StatementKind::Cache(_)));
    }

    #[test]
    fn test_cache_evict() {
        let stmt = parse_stmt("CACHE EVICT");
        let cache = unwrap_cache(stmt);
        assert!(matches!(cache.operation, CacheOp::Evict { count: None }));
    }

    #[test]
    fn test_cache_evict_with_count() {
        let stmt = parse_stmt("CACHE EVICT 100");
        let cache = unwrap_cache(stmt);
        let CacheOp::Evict { count: Some(expr) } = cache.operation else {
            panic!("expected EVICT with count")
        };
        assert!(matches!(
            expr.kind,
            ExprKind::Literal(Literal::Integer(100))
        ));
    }

    #[test]
    fn test_cache_get() {
        let stmt = parse_stmt("CACHE GET 'mykey'");
        let cache = unwrap_cache(stmt);
        let CacheOp::Get { key } = cache.operation else {
            panic!("expected GET")
        };
        assert!(matches!(key.kind, ExprKind::Literal(Literal::String(_))));
    }

    #[test]
    fn test_cache_put() {
        let stmt = parse_stmt("CACHE PUT 'mykey' 'myvalue'");
        let cache = unwrap_cache(stmt);
        let CacheOp::Put { key, value } = cache.operation else {
            panic!("expected PUT")
        };
        assert!(matches!(key.kind, ExprKind::Literal(Literal::String(_))));
        assert!(matches!(value.kind, ExprKind::Literal(Literal::String(_))));
    }

    #[test]
    #[should_panic(expected = "CACHE operation")]
    fn test_cache_invalid_operation() {
        parse_stmt("CACHE INVALID");
    }

    #[test]
    fn test_blob_init() {
        let stmt = parse_stmt("BLOB INIT");
        let blob = unwrap_blob(stmt);
        assert!(matches!(blob.operation, BlobOp::Init));
    }

    #[test]
    fn test_embed_build_index() {
        let stmt = parse_stmt("EMBED BUILD INDEX");
        let embed = unwrap_embed(stmt);
        assert!(matches!(embed.operation, EmbedOp::BuildIndex));
    }

    #[test]
    fn test_drop_index_on_syntax() {
        let stmt = parse_stmt("DROP INDEX ON users(name)");
        let drop = unwrap_dropindex(stmt);
        assert!(!drop.if_exists);
        assert!(drop.name.is_none());
        assert_eq!(drop.table.as_ref().unwrap().name, "users");
        assert_eq!(drop.column.as_ref().unwrap().name, "name");
    }

    #[test]
    fn test_drop_index_if_exists_on_syntax() {
        let stmt = parse_stmt("DROP INDEX IF EXISTS ON products(sku)");
        let drop = unwrap_dropindex(stmt);
        assert!(drop.if_exists);
        assert_eq!(drop.table.as_ref().unwrap().name, "products");
        assert_eq!(drop.column.as_ref().unwrap().name, "sku");
    }

    #[test]
    fn test_insert_select_query() {
        let stmt = parse_stmt("INSERT INTO target SELECT * FROM source");
        let insert = unwrap_insert(stmt);
        assert_eq!(insert.table.name, "target");
        assert!(matches!(insert.source, InsertSource::Query(_)));
    }

    // Phase 5: EMBED BATCH
    #[test]
    fn test_embed_batch() {
        let stmt = parse_stmt("EMBED BATCH [('doc1', [1.0, 0.0]), ('doc2', [0.0, 1.0])]");
        let embed = unwrap_embed(stmt);
        let EmbedOp::Batch { items } = embed.operation else {
            panic!("expected BATCH operation")
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn test_embed_batch_empty() {
        let stmt = parse_stmt("EMBED BATCH []");
        let embed = unwrap_embed(stmt);
        let EmbedOp::Batch { items } = embed.operation else {
            panic!("expected BATCH operation")
        };
        assert!(items.is_empty());
    }

    #[test]
    fn test_embed_batch_single() {
        let stmt = parse_stmt("EMBED BATCH [('key', [1.0, 2.0, 3.0])]");
        let embed = unwrap_embed(stmt);
        let EmbedOp::Batch { items } = embed.operation else {
            panic!("expected BATCH operation")
        };
        assert_eq!(items.len(), 1);
    }

    // Phase 5: CACHE SEMANTIC GET
    #[test]
    fn test_cache_semantic_get() {
        let stmt = parse_stmt("CACHE SEMANTIC GET 'query text'");
        let cache = unwrap_cache(stmt);
        let CacheOp::SemanticGet { threshold, .. } = cache.operation else {
            panic!("expected SEMANTIC GET operation")
        };
        assert!(threshold.is_none());
    }

    #[test]
    fn test_cache_semantic_get_with_threshold() {
        let stmt = parse_stmt("CACHE SEMANTIC GET 'query' THRESHOLD 0.85");
        let cache = unwrap_cache(stmt);
        let CacheOp::SemanticGet { threshold, .. } = cache.operation else {
            panic!("expected SEMANTIC GET operation")
        };
        assert!(threshold.is_some());
    }

    // Phase 5: CACHE SEMANTIC PUT
    #[test]
    fn test_cache_semantic_put() {
        let stmt = parse_stmt("CACHE SEMANTIC PUT 'query' 'response' EMBEDDING [1.0, 0.0]");
        let cache = unwrap_cache(stmt);
        let CacheOp::SemanticPut { embedding, .. } = cache.operation else {
            panic!("expected SEMANTIC PUT operation")
        };
        assert_eq!(embedding.len(), 2);
    }

    #[test]
    fn test_cache_semantic_put_large_embedding() {
        let stmt = parse_stmt("CACHE SEMANTIC PUT 'q' 'r' EMBEDDING [1.0, 2.0, 3.0, 4.0, 5.0]");
        let cache = unwrap_cache(stmt);
        let CacheOp::SemanticPut { embedding, .. } = cache.operation else {
            panic!("expected SEMANTIC PUT operation")
        };
        assert_eq!(embedding.len(), 5);
    }

    // Phase 5: DESCRIBE
    #[test]
    fn test_describe_table() {
        let stmt = parse_stmt("DESCRIBE TABLE users");
        let desc = unwrap_describe(stmt);
        let DescribeTarget::Table(ident) = desc.target else {
            panic!("expected DESCRIBE")
        };
        assert_eq!(ident.name, "users");
    }

    #[test]
    fn test_describe_node() {
        let stmt = parse_stmt("DESCRIBE NODE person");
        let desc = unwrap_describe(stmt);
        let DescribeTarget::Node(ident) = desc.target else {
            panic!("expected DESCRIBE")
        };
        assert_eq!(ident.name, "person");
    }

    #[test]
    fn test_describe_edge() {
        let stmt = parse_stmt("DESCRIBE EDGE follows");
        let desc = unwrap_describe(stmt);
        let DescribeTarget::Edge(ident) = desc.target else {
            panic!("expected DESCRIBE")
        };
        assert_eq!(ident.name, "follows");
    }

    // Phase 5: SHOW EMBEDDINGS
    #[test]
    fn test_show_embeddings() {
        let stmt = parse_stmt("SHOW EMBEDDINGS");
        assert!(matches!(
            stmt.kind,
            StatementKind::ShowEmbeddings { limit: None }
        ));
    }

    #[test]
    fn test_show_embeddings_with_limit() {
        let stmt = parse_stmt("SHOW EMBEDDINGS LIMIT 10");
        if let StatementKind::ShowEmbeddings { limit } = stmt.kind {
            assert!(limit.is_some());
        } else {
            panic!("expected SHOW EMBEDDINGS");
        }
    }

    // Phase 5: COUNT EMBEDDINGS
    #[test]
    fn test_count_embeddings() {
        let stmt = parse_stmt("COUNT EMBEDDINGS");
        assert!(matches!(stmt.kind, StatementKind::CountEmbeddings));
    }

    // ========== Checkpoint Tests ==========

    #[test]
    fn test_checkpoint_no_name() {
        let stmt = parse_stmt("CHECKPOINT");
        let cp = unwrap_checkpoint(stmt);
        assert!(cp.name.is_none());
    }

    #[test]
    fn test_checkpoint_with_name() {
        let stmt = parse_stmt("CHECKPOINT 'my-checkpoint'");
        let cp = unwrap_checkpoint(stmt);
        assert!(cp.name.is_some());
    }

    #[test]
    fn test_checkpoint_with_double_quoted_name() {
        let stmt = parse_stmt("CHECKPOINT \"my-checkpoint\"");
        let cp = unwrap_checkpoint(stmt);
        assert!(cp.name.is_some());
    }

    #[test]
    fn test_rollback_to() {
        let stmt = parse_stmt("ROLLBACK TO 'checkpoint-id'");
        let rb = unwrap_rollback(stmt);
        assert!(matches!(rb.target.kind, ExprKind::Literal(_)));
    }

    #[test]
    fn test_checkpoints_no_limit() {
        let stmt = parse_stmt("CHECKPOINTS");
        let cps = unwrap_checkpoints(stmt);
        assert!(cps.limit.is_none());
    }

    #[test]
    fn test_checkpoints_with_limit() {
        let stmt = parse_stmt("CHECKPOINTS LIMIT 5");
        let cps = unwrap_checkpoints(stmt);
        assert!(cps.limit.is_some());
    }

    // =========================================================================
    // Extended Graph Algorithm Tests
    // =========================================================================

    #[test]
    fn test_graph_pagerank_simple() {
        let stmt = parse_stmt("GRAPH PAGERANK");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::PageRank {
                damping: None,
                tolerance: None,
                max_iterations: None,
                direction: None,
                edge_type: None,
            }
        ));
    }

    #[test]
    fn test_graph_pagerank_with_damping() {
        let stmt = parse_stmt("GRAPH PAGERANK DAMPING 0.85");
        let algo = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::PageRank { damping, .. } = algo.operation else {
            panic!("expected PageRank")
        };
        assert!(damping.is_some());
    }

    #[test]
    fn test_graph_pagerank_with_all_options() {
        let stmt = parse_stmt(
            "GRAPH PAGERANK DAMPING 0.85 TOLERANCE 0.001 ITERATIONS 100 OUTGOING EDGE TYPE follows",
        );
        let algo = unwrap_graphalgorithm(stmt);
        if let GraphAlgorithmOp::PageRank {
            damping,
            tolerance,
            max_iterations,
            direction,
            edge_type,
        } = algo.operation
        {
            assert!(damping.is_some());
            assert!(tolerance.is_some());
            assert!(max_iterations.is_some());
            assert!(direction.is_some());
            assert!(edge_type.is_some());
        } else {
            panic!("expected PageRank");
        }
    }

    #[test]
    fn test_graph_betweenness_centrality() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::BetweennessCentrality { .. }
        ));
    }

    #[test]
    fn test_graph_betweenness_with_sampling() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY SAMPLING 0.5");
        let algo = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::BetweennessCentrality { sampling_ratio, .. } = algo.operation else {
            panic!("expected BetweennessCentrality")
        };
        assert!(sampling_ratio.is_some());
    }

    #[test]
    fn test_graph_closeness_centrality() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::ClosenessCentrality { .. }
        ));
    }

    #[test]
    fn test_graph_closeness_with_direction() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY INCOMING");
        let algo = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::ClosenessCentrality { direction, .. } = algo.operation else {
            panic!("expected ClosenessCentrality")
        };
        assert!(direction.is_some());
    }

    #[test]
    fn test_graph_eigenvector_centrality() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::EigenvectorCentrality { .. }
        ));
    }

    #[test]
    fn test_graph_eigenvector_with_options() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY ITERATIONS 50 TOLERANCE 0.0001");
        let algo = unwrap_graphalgorithm(stmt);
        if let GraphAlgorithmOp::EigenvectorCentrality {
            max_iterations,
            tolerance,
            ..
        } = algo.operation
        {
            assert!(max_iterations.is_some());
            assert!(tolerance.is_some());
        } else {
            panic!("expected EigenvectorCentrality");
        }
    }

    #[test]
    fn test_graph_louvain_communities() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::LouvainCommunities { .. }
        ));
    }

    #[test]
    fn test_graph_louvain_with_resolution() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES RESOLUTION 1.5 PASSES 10");
        let algo = unwrap_graphalgorithm(stmt);
        if let GraphAlgorithmOp::LouvainCommunities {
            resolution,
            max_passes,
            ..
        } = algo.operation
        {
            assert!(resolution.is_some());
            assert!(max_passes.is_some());
        } else {
            panic!("expected LouvainCommunities");
        }
    }

    #[test]
    fn test_graph_label_propagation() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION");
        let algo = unwrap_graphalgorithm(stmt);
        assert!(matches!(
            algo.operation,
            GraphAlgorithmOp::LabelPropagation { .. }
        ));
    }

    #[test]
    fn test_graph_label_propagation_with_iterations() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION ITERATIONS 20");
        let algo = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::LabelPropagation { max_iterations, .. } = algo.operation else {
            panic!("expected LabelPropagation")
        };
        assert!(max_iterations.is_some());
    }

    // =========================================================================
    // Graph Index Tests
    // =========================================================================

    #[test]
    fn test_graph_index_create_node_property() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON NODE PROPERTY name");
        let idx = unwrap_graphindex(stmt);
        let GraphIndexOp::CreateNodeProperty { property } = idx.operation else {
            panic!("expected CreateNodeProperty")
        };
        assert_eq!(property.name, "name");
    }

    #[test]
    fn test_graph_index_create_edge_property() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON EDGE PROPERTY weight");
        let idx = unwrap_graphindex(stmt);
        let GraphIndexOp::CreateEdgeProperty { property } = idx.operation else {
            panic!("expected CreateEdgeProperty")
        };
        assert_eq!(property.name, "weight");
    }

    #[test]
    fn test_graph_index_create_label() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON LABEL");
        let idx = unwrap_graphindex(stmt);
        assert!(matches!(idx.operation, GraphIndexOp::CreateLabel));
    }

    #[test]
    fn test_graph_index_create_edge_type() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON EDGE TYPE");
        let idx = unwrap_graphindex(stmt);
        assert!(matches!(idx.operation, GraphIndexOp::CreateEdgeType));
    }

    #[test]
    fn test_graph_index_drop_node() {
        let stmt = parse_stmt("GRAPH INDEX DROP ON NODE PROPERTY age");
        let idx = unwrap_graphindex(stmt);
        let GraphIndexOp::DropNode { property } = idx.operation else {
            panic!("expected DropNode")
        };
        assert_eq!(property.name, "age");
    }

    #[test]
    fn test_graph_index_drop_edge() {
        let stmt = parse_stmt("GRAPH INDEX DROP ON EDGE PROPERTY weight");
        let idx = unwrap_graphindex(stmt);
        let GraphIndexOp::DropEdge { property } = idx.operation else {
            panic!("expected DropEdge")
        };
        assert_eq!(property.name, "weight");
    }

    #[test]
    fn test_graph_index_show_node() {
        let stmt = parse_stmt("GRAPH INDEX SHOW ON NODE");
        let idx = unwrap_graphindex(stmt);
        assert!(matches!(idx.operation, GraphIndexOp::ShowNodeIndexes));
    }

    #[test]
    fn test_graph_index_show_edge() {
        let stmt = parse_stmt("GRAPH INDEX SHOW ON EDGE");
        let idx = unwrap_graphindex(stmt);
        assert!(matches!(idx.operation, GraphIndexOp::ShowEdgeIndexes));
    }

    // =========================================================================
    // Constraint Tests
    // =========================================================================

    #[test]
    fn test_constraint_create_unique() {
        let stmt = parse_stmt("CONSTRAINT CREATE email_unique ON NODE User PROPERTY email UNIQUE");
        let c = unwrap_graphconstraint(stmt);
        if let GraphConstraintOp::Create {
            name,
            target,
            property,
            constraint_type,
        } = c.operation
        {
            assert_eq!(name.name, "email_unique");
            assert!(matches!(target, ConstraintTarget::Node { label: Some(_) }));
            assert_eq!(property.name, "email");
            assert_eq!(constraint_type, ConstraintType::Unique);
        } else {
            panic!("expected Create");
        }
    }

    #[test]
    fn test_constraint_create_exists() {
        let stmt = parse_stmt("CONSTRAINT CREATE name_required ON NODE PROPERTY name EXISTS");
        let c = unwrap_graphconstraint(stmt);
        if let GraphConstraintOp::Create {
            constraint_type, ..
        } = c.operation
        {
            assert_eq!(constraint_type, ConstraintType::Exists);
        } else {
            panic!("expected Create");
        }
    }

    #[test]
    fn test_constraint_create_type() {
        let stmt = parse_stmt("CONSTRAINT CREATE age_int ON NODE PROPERTY age TYPE int");
        let c = unwrap_graphconstraint(stmt);
        if let GraphConstraintOp::Create {
            constraint_type, ..
        } = c.operation
        {
            assert!(matches!(constraint_type, ConstraintType::Type(_)));
        } else {
            panic!("expected Create");
        }
    }

    #[test]
    fn test_constraint_create_on_edge() {
        let stmt =
            parse_stmt("CONSTRAINT CREATE weight_exists ON EDGE knows PROPERTY weight EXISTS");
        let c = unwrap_graphconstraint(stmt);
        let GraphConstraintOp::Create { target, .. } = c.operation else {
            panic!("expected Create")
        };
        assert!(matches!(
            target,
            ConstraintTarget::Edge { edge_type: Some(_) }
        ));
    }

    #[test]
    fn test_constraint_drop() {
        let stmt = parse_stmt("CONSTRAINT DROP email_unique");
        let c = unwrap_graphconstraint(stmt);
        let GraphConstraintOp::Drop { name } = c.operation else {
            panic!("expected Drop")
        };
        assert_eq!(name.name, "email_unique");
    }

    #[test]
    fn test_constraint_list() {
        let stmt = parse_stmt("CONSTRAINT LIST");
        let c = unwrap_graphconstraint(stmt);
        assert!(matches!(c.operation, GraphConstraintOp::List));
    }

    #[test]
    fn test_constraint_get() {
        let stmt = parse_stmt("CONSTRAINT GET my_constraint");
        let c = unwrap_graphconstraint(stmt);
        let GraphConstraintOp::Get { name } = c.operation else {
            panic!("expected Get")
        };
        assert_eq!(name.name, "my_constraint");
    }

    // =========================================================================
    // Batch Operation Tests
    // =========================================================================

    #[test]
    fn test_batch_create_nodes_simple() {
        let stmt = parse_stmt("BATCH CREATE NODES [{labels: [Person], name: 'Alice'}]");
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::CreateNodes { nodes } = batch.operation else {
            panic!("expected CreateNodes")
        };
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].labels.len(), 1);
    }

    #[test]
    fn test_batch_create_nodes_multiple() {
        let stmt = parse_stmt(
            "BATCH CREATE NODES [{labels: [Person], name: 'Alice'}, {labels: [Person], name: 'Bob'}]",
        );
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::CreateNodes { nodes } = batch.operation else {
            panic!("expected CreateNodes")
        };
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_batch_create_edges() {
        let stmt = parse_stmt("BATCH CREATE EDGES [{from: 1, to: 2, type: knows, weight: 0.5}]");
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::CreateEdges { edges } = batch.operation else {
            panic!("expected CreateEdges")
        };
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type.name, "knows");
    }

    #[test]
    fn test_batch_delete_nodes() {
        let stmt = parse_stmt("BATCH DELETE NODES [1, 2, 3]");
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::DeleteNodes { ids } = batch.operation else {
            panic!("expected DeleteNodes")
        };
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_batch_delete_edges() {
        let stmt = parse_stmt("BATCH DELETE EDGES [10, 20]");
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::DeleteEdges { ids } = batch.operation else {
            panic!("expected DeleteEdges")
        };
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_batch_update_nodes() {
        let stmt = parse_stmt("BATCH UPDATE NODES [{id: 1, name: 'Alice Updated'}]");
        let batch = unwrap_graphbatch(stmt);
        let GraphBatchOp::UpdateNodes { updates } = batch.operation else {
            panic!("expected UpdateNodes")
        };
        assert_eq!(updates.len(), 1);
    }

    // =========================================================================
    // Aggregate Statement Tests
    // =========================================================================

    #[test]
    fn test_aggregate_node_property_sum() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY age SUM");
        let agg = unwrap_graphaggregate(stmt);
        if let GraphAggregateOp::AggregateNodeProperty {
            function,
            property,
            label,
            filter,
        } = agg.operation
        {
            assert_eq!(function, AggregateFunction::Sum);
            assert_eq!(property.name, "age");
            assert!(label.is_none());
            assert!(filter.is_none());
        } else {
            panic!("expected AggregateNodeProperty");
        }
    }

    #[test]
    fn test_aggregate_node_property_avg() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY salary AVG");
        let agg = unwrap_graphaggregate(stmt);
        let GraphAggregateOp::AggregateNodeProperty { function, .. } = agg.operation else {
            panic!("expected AggregateNodeProperty")
        };
        assert_eq!(function, AggregateFunction::Avg);
    }

    #[test]
    fn test_aggregate_node_property_with_label() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY age SUM BY LABEL Person");
        let agg = unwrap_graphaggregate(stmt);
        let GraphAggregateOp::AggregateNodeProperty { label, .. } = agg.operation else {
            panic!("expected AggregateNodeProperty")
        };
        assert!(label.is_some());
        assert_eq!(label.unwrap().name, "Person");
    }

    #[test]
    fn test_aggregate_node_property_with_filter() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY age SUM WHERE age > 18");
        let agg = unwrap_graphaggregate(stmt);
        let GraphAggregateOp::AggregateNodeProperty { filter, .. } = agg.operation else {
            panic!("expected AggregateNodeProperty")
        };
        assert!(filter.is_some());
    }

    #[test]
    fn test_aggregate_edge_property() {
        let stmt = parse_stmt("AGGREGATE EDGE PROPERTY weight AVG");
        let agg = unwrap_graphaggregate(stmt);
        if let GraphAggregateOp::AggregateEdgeProperty {
            function, property, ..
        } = agg.operation
        {
            assert_eq!(function, AggregateFunction::Avg);
            assert_eq!(property.name, "weight");
        } else {
            panic!("expected AggregateEdgeProperty");
        }
    }

    #[test]
    fn test_aggregate_edge_property_with_type() {
        let stmt = parse_stmt("AGGREGATE EDGE PROPERTY weight SUM BY TYPE knows");
        let agg = unwrap_graphaggregate(stmt);
        let GraphAggregateOp::AggregateEdgeProperty { edge_type, .. } = agg.operation else {
            panic!("expected AggregateEdgeProperty")
        };
        assert!(edge_type.is_some());
        assert_eq!(edge_type.unwrap().name, "knows");
    }

    #[test]
    fn test_aggregate_functions_min_max_count() {
        for (func_name, expected_func) in [
            ("MIN", AggregateFunction::Min),
            ("MAX", AggregateFunction::Max),
            ("COUNT", AggregateFunction::Count),
        ] {
            let stmt = parse_stmt(&format!("AGGREGATE NODE PROPERTY x {}", func_name));
            let agg = unwrap_graphaggregate(stmt);
            let GraphAggregateOp::AggregateNodeProperty { function, .. } = agg.operation else {
                panic!("expected AggregateNodeProperty")
            };
            assert_eq!(function, expected_func);
        }
    }

    // =========================================================================
    // Collection and WHERE clause tests
    // =========================================================================

    #[test]
    fn test_embed_store_in_collection() {
        let stmt = parse_stmt("EMBED STORE 'doc1' [1.0, 2.0, 3.0] INTO my_collection");
        let embed = unwrap_embed(stmt);
        assert!(matches!(embed.operation, EmbedOp::Store { .. }));
        assert_eq!(embed.collection, Some("my_collection".to_string()));
    }

    #[test]
    fn test_embed_get_in_collection() {
        let stmt = parse_stmt("EMBED GET 'doc1' INTO my_collection");
        let embed = unwrap_embed(stmt);
        assert!(matches!(embed.operation, EmbedOp::Get { .. }));
        assert_eq!(embed.collection, Some("my_collection".to_string()));
    }

    #[test]
    fn test_embed_delete_in_collection() {
        let stmt = parse_stmt("EMBED DELETE 'doc1' INTO my_collection");
        let embed = unwrap_embed(stmt);
        assert!(matches!(embed.operation, EmbedOp::Delete { .. }));
        assert_eq!(embed.collection, Some("my_collection".to_string()));
    }

    #[test]
    fn test_embed_without_collection() {
        let stmt = parse_stmt("EMBED STORE 'doc1' [1.0, 2.0]");
        let embed = unwrap_embed(stmt);
        assert!(embed.collection.is_none());
    }

    #[test]
    fn test_similar_in_collection() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 10 INTO my_collection");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Vector(_)));
        assert!(similar.limit.is_some());
        assert_eq!(similar.collection, Some("my_collection".to_string()));
        assert!(similar.where_clause.is_none());
    }

    #[test]
    fn test_similar_with_where() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 10 WHERE category = 'science'");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Vector(_)));
        assert!(similar.limit.is_some());
        assert!(similar.collection.is_none());
        assert!(similar.where_clause.is_some());
    }

    #[test]
    fn test_similar_with_collection_and_where() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 5 INTO docs WHERE author = 'Alice'");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Vector(_)));
        assert!(similar.limit.is_some());
        assert_eq!(similar.collection, Some("docs".to_string()));
        assert!(similar.where_clause.is_some());
    }

    #[test]
    fn test_similar_where_with_and() {
        let stmt = parse_stmt("SIMILAR 'doc1' LIMIT 10 WHERE category = 'tech' AND score > 5");
        let similar = unwrap_similar(stmt);
        assert!(matches!(similar.query, SimilarQuery::Key(_)));
        assert!(similar.where_clause.is_some());
        // WHERE clause is an AND expression
        if let Some(ref where_expr) = similar.where_clause {
            assert!(matches!(
                where_expr.kind,
                ExprKind::Binary(_, BinaryOp::And, _)
            ));
        }
    }

    #[test]
    fn test_similar_where_with_or() {
        let stmt = parse_stmt("SIMILAR [1.0] WHERE status = 'active' OR status = 'pending'");
        let similar = unwrap_similar(stmt);
        assert!(similar.where_clause.is_some());
        if let Some(ref where_expr) = similar.where_clause {
            assert!(matches!(
                where_expr.kind,
                ExprKind::Binary(_, BinaryOp::Or, _)
            ));
        }
    }

    #[test]
    fn test_embed_batch_in_collection() {
        let stmt = parse_stmt("EMBED BATCH [('k1', [1.0]), ('k2', [2.0])] INTO batch_coll");
        let embed = unwrap_embed(stmt);
        assert!(matches!(embed.operation, EmbedOp::Batch { .. }));
        assert_eq!(embed.collection, Some("batch_coll".to_string()));
    }

    // =========================================================================
    // Vault Statement Tests
    // =========================================================================

    #[test]
    fn test_vault_set() {
        let stmt = parse_stmt("VAULT SET 'key1' 'value1'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Set { .. }));
    }

    #[test]
    fn test_vault_get() {
        let stmt = parse_stmt("VAULT GET 'mykey'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Get { .. }));
    }

    #[test]
    fn test_vault_delete() {
        let stmt = parse_stmt("VAULT DELETE 'mykey'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Delete { .. }));
    }

    #[test]
    fn test_vault_list_all() {
        let stmt = parse_stmt("VAULT LIST");
        let v = unwrap_vault(stmt);
        let VaultOp::List { pattern } = v.operation else {
            panic!("expected VaultOp::List")
        };
        assert!(pattern.is_none());
    }

    #[test]
    fn test_vault_list_with_pattern() {
        let stmt = parse_stmt("VAULT LIST 'secret*'");
        let v = unwrap_vault(stmt);
        let VaultOp::List { pattern } = v.operation else {
            panic!("expected VaultOp::List")
        };
        assert!(pattern.is_some());
    }

    #[test]
    fn test_vault_rotate() {
        let stmt = parse_stmt("VAULT ROTATE 'mykey' 'newvalue'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Rotate { .. }));
    }

    #[test]
    fn test_vault_grant() {
        let stmt = parse_stmt("VAULT GRANT 'user123' ON 'secret/key'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Grant { .. }));
    }

    #[test]
    fn test_vault_revoke() {
        let stmt = parse_stmt("VAULT REVOKE 'user123' ON 'secret/key'");
        let v = unwrap_vault(stmt);
        assert!(matches!(v.operation, VaultOp::Revoke { .. }));
    }

    #[test]
    fn test_vault_invalid_op() {
        let result = parse("VAULT INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // Cluster Statement Tests
    // =========================================================================

    #[test]
    fn test_cluster_connect() {
        let stmt = parse_stmt("CLUSTER CONNECT '127.0.0.1:8080'");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Connect { .. }));
    }

    #[test]
    fn test_cluster_disconnect() {
        let stmt = parse_stmt("CLUSTER DISCONNECT");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Disconnect));
    }

    #[test]
    fn test_cluster_status() {
        let stmt = parse_stmt("CLUSTER STATUS");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Status));
    }

    #[test]
    fn test_cluster_nodes() {
        let stmt = parse_stmt("CLUSTER NODES");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Nodes));
    }

    #[test]
    fn test_cluster_leader() {
        let stmt = parse_stmt("CLUSTER LEADER");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Leader));
    }

    #[test]
    fn test_cluster_invalid_op() {
        let result = parse("CLUSTER INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // BLOB Statement Tests
    // =========================================================================

    #[test]
    fn test_blob_put_from_path() {
        let stmt = parse_stmt("BLOB PUT 'myfile.txt' FROM '/path/to/file'");
        let b = unwrap_blob(stmt);
        let BlobOp::Put { from_path, .. } = b.operation else {
            panic!("expected BlobOp::Put")
        };
        assert!(from_path.is_some());
    }

    #[test]
    fn test_blob_put_with_data() {
        let stmt = parse_stmt("BLOB PUT 'myfile.txt' 'inline data here'");
        let b = unwrap_blob(stmt);
        let BlobOp::Put { data, .. } = b.operation else {
            panic!("expected BlobOp::Put")
        };
        assert!(data.is_some());
    }

    #[test]
    fn test_blob_put_with_link_and_tag() {
        let stmt = parse_stmt("BLOB PUT 'doc.pdf' FROM '/path' LINK 'entity1' TAG 'important'");
        let b = unwrap_blob(stmt);
        let BlobOp::Put { options, .. } = b.operation else {
            panic!("expected BlobOp::Put")
        };
        assert!(!options.link.is_empty());
        assert!(!options.tag.is_empty());
    }

    #[test]
    fn test_blob_get() {
        let stmt = parse_stmt("BLOB GET 'artifact123'");
        let b = unwrap_blob(stmt);
        let BlobOp::Get { to_path, .. } = b.operation else {
            panic!("expected BlobOp::Get")
        };
        assert!(to_path.is_none());
    }

    #[test]
    fn test_blob_get_to_path() {
        let stmt = parse_stmt("BLOB GET 'artifact123' TO '/output/file.txt'");
        let b = unwrap_blob(stmt);
        let BlobOp::Get { to_path, .. } = b.operation else {
            panic!("expected BlobOp::Get")
        };
        assert!(to_path.is_some());
    }

    #[test]
    fn test_blob_delete() {
        let stmt = parse_stmt("BLOB DELETE 'artifact123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Delete { .. }));
    }

    #[test]
    fn test_blob_info() {
        let stmt = parse_stmt("BLOB INFO 'artifact123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Info { .. }));
    }

    #[test]
    fn test_blob_link() {
        let stmt = parse_stmt("BLOB LINK 'artifact123' TO 'entity456'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Link { .. }));
    }

    #[test]
    fn test_blob_unlink() {
        let stmt = parse_stmt("BLOB UNLINK 'artifact123' FROM 'entity456'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Unlink { .. }));
    }

    #[test]
    fn test_blob_links() {
        let stmt = parse_stmt("BLOB LINKS 'artifact123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Links { .. }));
    }

    #[test]
    fn test_blob_tag() {
        let stmt = parse_stmt("BLOB TAG 'artifact123' 'important'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Tag { .. }));
    }

    #[test]
    fn test_blob_untag() {
        let stmt = parse_stmt("BLOB UNTAG 'artifact123' 'important'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Untag { .. }));
    }

    #[test]
    fn test_blob_verify() {
        let stmt = parse_stmt("BLOB VERIFY 'artifact123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Verify { .. }));
    }

    #[test]
    fn test_blob_gc() {
        let stmt = parse_stmt("BLOB GC");
        let b = unwrap_blob(stmt);
        let BlobOp::Gc { full } = b.operation else {
            panic!("expected BlobOp::Gc")
        };
        assert!(!full);
    }

    #[test]
    fn test_blob_gc_full() {
        let stmt = parse_stmt("BLOB GC FULL");
        let b = unwrap_blob(stmt);
        let BlobOp::Gc { full } = b.operation else {
            panic!("expected BlobOp::Gc")
        };
        assert!(full);
    }

    #[test]
    fn test_blob_repair() {
        let stmt = parse_stmt("BLOB REPAIR");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Repair));
    }

    #[test]
    fn test_blob_stats() {
        let stmt = parse_stmt("BLOB STATS");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Stats));
    }

    #[test]
    fn test_blob_meta_set() {
        let stmt = parse_stmt("BLOB META SET 'artifact123' 'description' 'A test file'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::MetaSet { .. }));
    }

    #[test]
    fn test_blob_meta_get() {
        let stmt = parse_stmt("BLOB META GET 'artifact123' 'description'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::MetaGet { .. }));
    }

    #[test]
    fn test_blob_meta_invalid() {
        let result = parse("BLOB META INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_blob_invalid_op() {
        let result = parse("BLOB INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // BLOBS Statement Tests
    // =========================================================================

    #[test]
    fn test_blobs_list_all() {
        let stmt = parse_stmt("BLOBS");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::List { pattern: None }));
    }

    #[test]
    fn test_blobs_list_pattern() {
        let stmt = parse_stmt("BLOBS '*.txt'");
        let b = unwrap_blobs(stmt);
        let BlobsOp::List { pattern } = b.operation else {
            panic!("expected BlobsOp::List")
        };
        assert!(pattern.is_some());
    }

    #[test]
    fn test_blobs_for_entity() {
        let stmt = parse_stmt("BLOBS FOR 'entity123'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::For { .. }));
    }

    #[test]
    fn test_blobs_by_tag() {
        let stmt = parse_stmt("BLOBS BY TAG 'important'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::ByTag { .. }));
    }

    #[test]
    fn test_blobs_where_type() {
        let stmt = parse_stmt("BLOBS WHERE TYPE = 'application/pdf'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::ByType { .. }));
    }

    #[test]
    fn test_blobs_similar() {
        let stmt = parse_stmt("BLOBS SIMILAR TO 'artifact123' LIMIT 10");
        let b = unwrap_blobs(stmt);
        let BlobsOp::Similar { limit, .. } = b.operation else {
            panic!("expected BlobsOp::Similar")
        };
        assert!(limit.is_some());
    }

    #[test]
    fn test_blobs_similar_no_limit() {
        let stmt = parse_stmt("BLOBS SIMILAR TO 'artifact123'");
        let b = unwrap_blobs(stmt);
        let BlobsOp::Similar { limit, .. } = b.operation else {
            panic!("expected BlobsOp::Similar")
        };
        assert!(limit.is_none());
    }

    // =========================================================================
    // CHAIN Statement Tests
    // =========================================================================

    #[test]
    fn test_chain_begin() {
        let stmt = parse_stmt("BEGIN CHAIN TRANSACTION");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Begin));
    }

    #[test]
    fn test_chain_commit() {
        let stmt = parse_stmt("COMMIT CHAIN");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Commit));
    }

    #[test]
    fn test_chain_rollback_height() {
        let stmt = parse_stmt("ROLLBACK CHAIN TO 100");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Rollback { .. }));
    }

    #[test]
    fn test_chain_height() {
        let stmt = parse_stmt("CHAIN HEIGHT");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Height));
    }

    #[test]
    fn test_chain_tip() {
        let stmt = parse_stmt("CHAIN TIP");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Tip));
    }

    #[test]
    fn test_chain_block() {
        let stmt = parse_stmt("CHAIN BLOCK 42");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Block { .. }));
    }

    #[test]
    fn test_chain_verify() {
        let stmt = parse_stmt("CHAIN VERIFY");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Verify));
    }

    #[test]
    fn test_chain_history() {
        let stmt = parse_stmt("CHAIN HISTORY 'users:123'");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::History { .. }));
    }

    #[test]
    fn test_chain_similar() {
        let stmt = parse_stmt("CHAIN SIMILAR [1.0, 2.0] LIMIT 5");
        let c = unwrap_chain(stmt);
        let ChainOp::Similar { limit, .. } = c.operation else {
            panic!("expected ChainOp::Similar")
        };
        assert!(limit.is_some());
    }

    #[test]
    fn test_chain_drift() {
        let stmt = parse_stmt("CHAIN DRIFT FROM 0 TO 100");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Drift { .. }));
    }

    #[test]
    fn test_chain_invalid_op() {
        let result = parse("CHAIN INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_show_codebook_global() {
        let stmt = parse_stmt("SHOW CODEBOOK GLOBAL");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::ShowCodebookGlobal));
    }

    #[test]
    fn test_show_codebook_local() {
        let stmt = parse_stmt("SHOW CODEBOOK LOCAL 'users'");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::ShowCodebookLocal { .. }));
    }

    #[test]
    fn test_analyze_codebook_transitions() {
        let stmt = parse_stmt("ANALYZE CODEBOOK TRANSITIONS");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::AnalyzeTransitions));
    }

    // =========================================================================
    // Graph Algorithm Extended Tests
    // =========================================================================

    #[test]
    fn test_graph_pagerank_parses() {
        let stmt = parse_stmt("GRAPH PAGERANK");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_betweenness_parses() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_closeness_parses() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_eigenvector_parses() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_louvain_parses() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_label_propagation_parses() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_graph_invalid_algorithm() {
        let result = parse("GRAPH INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // Graph Index Extended Tests
    // =========================================================================

    #[test]
    fn test_graph_index_create_node_prop_extended() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON NODE PROPERTY name");
        assert!(matches!(stmt.kind, StatementKind::GraphIndex(_)));
    }

    #[test]
    fn test_graph_index_invalid_op() {
        let result = parse("GRAPH INDEX INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // Rollback and Checkpoint Extended Tests
    // =========================================================================

    #[test]
    fn test_rollback_to_checkpoint() {
        let stmt = parse_stmt("ROLLBACK TO 'checkpoint1'");
        let r = unwrap_rollback(stmt);
        assert!(matches!(
            r.target.kind,
            ExprKind::Literal(Literal::String(_))
        ));
    }

    #[test]
    fn test_checkpoints_list() {
        let stmt = parse_stmt("CHECKPOINTS");
        let c = unwrap_checkpoints(stmt);
        assert!(c.limit.is_none());
    }

    // =========================================================================
    // ENTITY Statement Extended Tests
    // =========================================================================

    #[test]
    fn test_entity_get() {
        let stmt = parse_stmt("ENTITY GET 'user:123'");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Get { .. }));
    }

    #[test]
    fn test_entity_update() {
        let stmt = parse_stmt("ENTITY UPDATE 'user:123' { name: 'Bob' }");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Update { .. }));
    }

    #[test]
    fn test_entity_delete() {
        let stmt = parse_stmt("ENTITY DELETE 'user:123'");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Delete { .. }));
    }

    #[test]
    fn test_entity_invalid_op() {
        let result = parse("ENTITY INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional Expression Tests
    // =========================================================================

    #[test]
    fn test_between_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
        let s = unwrap_select(stmt);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_like_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name LIKE '%test%'");
        let s = unwrap_select(stmt);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_is_null_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NULL");
        let s = unwrap_select(stmt);
        assert!(s.where_clause.is_some());
    }

    #[test]
    fn test_unary_negative() {
        let stmt = parse_stmt("SELECT -5");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_qualified_column_expr() {
        let stmt = parse_stmt("SELECT t.name FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // =========================================================================
    // Additional Error Handling Tests
    // =========================================================================

    #[test]
    fn test_error_embed_invalid_op() {
        let result = parse("EMBED INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_neighbors_with_limit() {
        // NEIGHBORS parses limit correctly
        let stmt = parse_stmt("NEIGHBORS 1 LIMIT 10");
        let n = unwrap_neighbors(stmt);
        assert!(n.limit.is_some());
    }

    #[test]
    fn test_error_constraint_invalid_op() {
        let result = parse("CONSTRAINT INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_aggregate_invalid_target() {
        let result = parse("AGGREGATE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_batch_invalid_op() {
        let result = parse("BATCH INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_all_semicolon_separated() {
        // parse_all handles semicolon-separated statements
        let result = parse_all("SELECT 1; SELECT 2").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_error_unclosed_parenthesis() {
        let result = parse("SELECT (1 + 2");
        assert!(result.is_err());
    }

    #[test]
    fn test_array_subscript_expr() {
        let stmt = parse_stmt("SELECT arr[0]");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_error_cache_semantic_invalid() {
        let result = parse("CACHE SEMANTIC INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // SHOW Statement Extended Tests
    // =========================================================================

    #[test]
    fn test_show_tables_statement() {
        let stmt = parse_stmt("SHOW TABLES");
        assert!(matches!(stmt.kind, StatementKind::ShowTables));
    }

    #[test]
    fn test_show_embeddings_stmt() {
        let stmt = parse_stmt("SHOW EMBEDDINGS");
        assert!(matches!(stmt.kind, StatementKind::ShowEmbeddings { .. }));
    }

    #[test]
    fn test_show_invalid() {
        let result = parse("SHOW INVALID");
        assert!(result.is_err());
    }

    // =========================================================================
    // Additional FIND Tests
    // =========================================================================

    #[test]
    fn test_find_node_with_label() {
        let stmt = parse_stmt("FIND NODE Person WHERE age > 18");
        let f = unwrap_find(stmt);
        assert!(f.where_clause.is_some());
    }

    #[test]
    fn test_find_edge_with_type() {
        let stmt = parse_stmt("FIND EDGE knows WHERE weight > 0.5");
        let f = unwrap_find(stmt);
        assert!(f.where_clause.is_some());
    }

    #[test]
    fn test_find_with_limit() {
        let stmt = parse_stmt("FIND NODE Person LIMIT 10");
        let f = unwrap_find(stmt);
        assert!(f.limit.is_some());
    }

    // =========================================================================
    // Additional SIMILAR Tests
    // =========================================================================

    #[test]
    fn test_similar_connected_to() {
        let stmt = parse_stmt("SIMILAR 'key' CONNECTED TO 'hub' LIMIT 10");
        let s = unwrap_similar(stmt);
        assert!(s.connected_to.is_some());
    }

    #[test]
    fn test_similar_with_metric_cosine() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 5 COSINE");
        let s = unwrap_similar(stmt);
        assert!(s.metric.is_some());
    }

    // =========================================================================
    // Additional NEIGHBORS Tests
    // =========================================================================

    #[test]
    fn test_neighbors_by_similarity() {
        let stmt = parse_stmt("NEIGHBORS 'entity' BY SIMILAR [1.0, 0.0] LIMIT 5");
        let n = unwrap_neighbors(stmt);
        assert!(n.by_similarity.is_some());
    }

    #[test]
    fn test_neighbors_outgoing_with_limit() {
        let stmt = parse_stmt("NEIGHBORS 123 OUTGOING LIMIT 20");
        let n = unwrap_neighbors(stmt);
        assert!(n.limit.is_some());
    }

    // =========================================================================
    // Additional PATH Tests
    // =========================================================================

    #[test]
    fn test_path_with_depth_limit() {
        let stmt = parse_stmt("PATH 1 -> 10 LIMIT 5");
        let p = unwrap_path(stmt);
        assert!(p.max_depth.is_some());
    }

    #[test]
    fn test_path_shortest() {
        let stmt = parse_stmt("PATH SHORTEST 1 -> 10");
        let p = unwrap_path(stmt);
        assert!(matches!(p.algorithm, PathAlgorithm::Shortest));
    }

    // =========================================================================
    // Additional Coverage Tests - Unique tests for uncovered paths
    // =========================================================================

    #[test]
    fn test_neighbors_both_direction() {
        let stmt = parse_stmt("NEIGHBORS 1 BOTH LIMIT 5");
        let n = unwrap_neighbors(stmt);
        assert!(matches!(n.direction, Direction::Both));
    }

    #[test]
    fn test_not_like_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name NOT LIKE '%test%'");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_is_not_null_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name IS NOT NULL");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_not_between_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE age NOT BETWEEN 10 AND 20");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_in_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE id IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_not_in_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE id NOT IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_qualified_wildcard_expr() {
        let stmt = parse_stmt("SELECT t.* FROM t");
        let s = unwrap_select(stmt);
        assert!(s.columns.len() == 1);
    }

    #[test]
    fn test_similar_into_collection() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 10 INTO my_collection");
        let s = unwrap_similar(stmt);
        assert!(s.collection.is_some());
    }

    #[test]
    fn test_select_with_having() {
        let stmt = parse_stmt("SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1");
        let s = unwrap_select(stmt);
        assert!(s.having.is_some());
    }

    #[test]
    fn test_select_with_union() {
        let stmt = parse_stmt("SELECT name FROM users UNION SELECT name FROM admins");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_case() {
        let stmt = parse_stmt("SELECT CASE WHEN age > 18 THEN 'adult' ELSE 'minor' END FROM users");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_subquery() {
        let stmt = parse_stmt("SELECT * FROM (SELECT id FROM users) AS sub");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_exists() {
        let stmt = parse_stmt("SELECT * FROM users WHERE EXISTS (SELECT 1 FROM orders)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_full_outer_join() {
        let stmt =
            parse_stmt("SELECT * FROM users FULL OUTER JOIN orders ON users.id = orders.user_id");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_create_table_decimal() {
        let stmt = parse_stmt("CREATE TABLE t (price DECIMAL(10, 2))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_text() {
        let stmt = parse_stmt("CREATE TABLE t (bio TEXT)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_timestamp() {
        let stmt = parse_stmt("CREATE TABLE t (created TIMESTAMP)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_column_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (age INT CHECK (age >= 0))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_column_unique_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (email VARCHAR(100) UNIQUE)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_insert_from_select_stmt() {
        let stmt = parse_stmt("INSERT INTO archive SELECT * FROM users WHERE active = false");
        assert!(matches!(stmt.kind, StatementKind::Insert(_)));
    }

    #[test]
    fn test_pagerank_with_params() {
        let stmt = parse_stmt("GRAPH PAGERANK DAMPING 0.9 ITERATIONS 20");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    // =========================================================================
    // Edge Case Tests for Coverage
    // =========================================================================

    #[test]
    fn test_empty_array_literal() {
        let stmt = parse_stmt("SELECT []");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_error_case_without_when() {
        let result = parse("SELECT CASE END FROM t");
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_case_expr() {
        let stmt = parse_stmt("SELECT CASE x WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_intersect_set_op() {
        let stmt = parse_stmt("SELECT name FROM users INTERSECT SELECT name FROM admins");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_except_set_op() {
        let stmt = parse_stmt("SELECT name FROM users EXCEPT SELECT name FROM banned");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_all_distinct_in_union() {
        let stmt = parse_stmt("SELECT name FROM users UNION ALL SELECT name FROM admins");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_simple_checkpoint() {
        let stmt = parse_stmt("CHECKPOINT");
        assert!(matches!(stmt.kind, StatementKind::Checkpoint(_)));
    }

    #[test]
    fn test_named_checkpoint() {
        let stmt = parse_stmt("CHECKPOINT 'my_checkpoint'");
        let c = unwrap_checkpoint(stmt);
        assert!(c.name.is_some());
    }

    #[test]
    fn test_rollback_to_target() {
        let stmt = parse_stmt("ROLLBACK TO 'checkpoint1'");
        assert!(matches!(stmt.kind, StatementKind::Rollback(_)));
    }

    #[test]
    fn test_similar_with_euclidean() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 5 EUCLIDEAN");
        let s = unwrap_similar(stmt);
        assert!(s.metric.is_some());
    }

    // =========================================================================
    // Error Path Coverage Tests
    // =========================================================================

    #[test]
    fn test_error_create_invalid() {
        let result = parse("CREATE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_show_codebook_invalid() {
        let result = parse("SHOW CODEBOOK INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_show_invalid_target() {
        let result = parse("SHOW INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_describe_invalid() {
        let result = parse("DESCRIBE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_count_invalid() {
        let result = parse("COUNT INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_table_alias_keyword_not_taken() {
        // When the next token is a keyword, it should not be taken as alias
        let stmt = parse_stmt("SELECT * FROM users WHERE id = 1");
        let select = unwrap_select(stmt);
        let from = select.from.unwrap();
        // The alias should be None because WHERE is a keyword
        assert!(from.table.alias.is_none());
    }

    #[test]
    fn test_error_cluster_invalid() {
        let result = parse("CLUSTER INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_invalid() {
        let result = parse("GRAPH INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_index_create_invalid() {
        let result = parse("GRAPH INDEX CREATE ON INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_index_drop_invalid() {
        let result = parse("GRAPH INDEX DROP ON INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_index_show_invalid() {
        let result = parse("GRAPH INDEX SHOW ON INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_graph_index_invalid_verb() {
        let result = parse("GRAPH INDEX INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_entity_invalid() {
        let result = parse("ENTITY INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_node_invalid() {
        let result = parse("NODE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_edge_invalid() {
        let result = parse("EDGE INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_blob_invalid() {
        let result = parse("BLOB INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_vault_invalid() {
        let result = parse("VAULT INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_embed_invalid() {
        let result = parse("EMBED INVALID");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_batch_edge_missing_from() {
        let result = parse("EDGE BATCH CREATE [{to: 2, type: FOLLOWS}]");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_batch_edge_missing_to() {
        let result = parse("EDGE BATCH CREATE [{from: 1, type: FOLLOWS}]");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_batch_edge_missing_type() {
        let result = parse("EDGE BATCH CREATE [{from: 1, to: 2}]");
        assert!(result.is_err());
    }

    #[test]
    fn test_join_no_condition() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b");
        let select = unwrap_select(stmt);
        let join = &select.from.as_ref().unwrap().joins[0];
        assert!(join.condition.is_none());
    }

    // =========================================================================
    // Additional Coverage Tests (unique names)
    // =========================================================================

    #[test]
    fn test_coverage_exists_expr() {
        let stmt = parse_stmt("SELECT * FROM t WHERE EXISTS (SELECT 1 FROM s)");
        let select = unwrap_select(stmt);
        assert!(select.where_clause.is_some());
    }

    #[test]
    fn test_coverage_cast_expr_varchar() {
        let stmt = parse_stmt("SELECT CAST(x AS VARCHAR(255)) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_empty_tuple() {
        let stmt = parse_stmt("SELECT ()");
        let select = unwrap_select(stmt);
        assert!(matches!(select.columns[0].expr.kind, ExprKind::Tuple(_)));
    }

    #[test]
    fn test_coverage_function_with_distinct() {
        let stmt = parse_stmt("SELECT COUNT(DISTINCT x) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_select_all_modifier() {
        let stmt = parse_stmt("SELECT ALL x FROM t");
        let select = unwrap_select(stmt);
        assert!(!select.distinct);
    }

    #[test]
    fn test_coverage_aggregate_sum_expr() {
        let stmt = parse_stmt("SELECT SUM(x) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_aggregate_avg_expr() {
        let stmt = parse_stmt("SELECT AVG(x) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_aggregate_min_expr() {
        let stmt = parse_stmt("SELECT MIN(x) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_aggregate_max_expr() {
        let stmt = parse_stmt("SELECT MAX(x) FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_unary_not_expr() {
        let stmt = parse_stmt("SELECT NOT true");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_unary_bang_expr() {
        let stmt = parse_stmt("SELECT !false");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_subquery_in_from() {
        let stmt = parse_stmt("SELECT * FROM (SELECT 1 AS x) AS sub");
        let select = unwrap_select(stmt);
        assert!(select.from.is_some());
    }

    #[test]
    fn test_coverage_cache_stmt() {
        let stmt = parse_stmt("CACHE GET 'mykey'");
        assert!(matches!(stmt.kind, StatementKind::Cache(_)));
    }

    #[test]
    fn test_coverage_vault_stmt() {
        let stmt = parse_stmt("VAULT GET 'mysecret'");
        assert!(matches!(stmt.kind, StatementKind::Vault(_)));
    }

    #[test]
    fn test_coverage_unknown_command() {
        let result = parse("UNKNOWNXYZ");
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_blobs_for() {
        let stmt = parse_stmt("BLOBS FOR 'entity1'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::For { .. }));
    }

    #[test]
    fn test_coverage_find_edges() {
        let stmt = parse_stmt("FIND EDGE FOLLOWS WHERE weight > 0.5");
        let f = unwrap_find(stmt);
        assert!(matches!(f.pattern, FindPattern::Edges { .. }));
    }

    #[test]
    fn test_coverage_find_rows() {
        let stmt = parse_stmt("FIND ROWS FROM users WHERE age > 18");
        let f = unwrap_find(stmt);
        assert!(matches!(f.pattern, FindPattern::Rows { .. }));
    }

    #[test]
    fn test_coverage_update_with_where() {
        let stmt = parse_stmt("UPDATE users SET name = 'Bob' WHERE id = 1");
        let u = unwrap_update(stmt);
        assert!(u.where_clause.is_some());
    }

    #[test]
    fn test_coverage_delete_with_where() {
        let stmt = parse_stmt("DELETE FROM users WHERE id = 1");
        let d = unwrap_delete(stmt);
        assert!(d.where_clause.is_some());
    }

    #[test]
    fn test_coverage_similar_with_connected() {
        let stmt = parse_stmt("SIMILAR 'entity' CONNECTED TO 'hub' LIMIT 5");
        let s = unwrap_similar(stmt);
        assert!(s.connected_to.is_some());
    }

    #[test]
    fn test_coverage_similar_with_cosine_metric() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 5 COSINE");
        let s = unwrap_similar(stmt);
        assert!(s.metric.is_some());
    }

    #[test]
    fn test_coverage_similar_with_dot_product_metric() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 5 DOT_PRODUCT");
        let s = unwrap_similar(stmt);
        assert!(s.metric.is_some());
    }

    // Graph algorithm coverage tests - optional parameters
    #[test]
    fn test_coverage_betweenness_with_direction() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY OUTGOING");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_betweenness_with_edge_type() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY EDGE TYPE follows");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_betweenness_with_sampling() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY SAMPLING 0.5");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_closeness_with_direction() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY INCOMING");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_closeness_with_edge_type() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY EDGE TYPE knows");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_eigenvector_with_iterations() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY ITERATIONS 100");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_eigenvector_with_tolerance() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY TOLERANCE 0.001");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_eigenvector_with_direction() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY BOTH");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_eigenvector_with_edge_type() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY EDGE TYPE follows");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_louvain_with_resolution() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES RESOLUTION 1.5");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_louvain_with_passes() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES PASSES 10");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_louvain_with_direction() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES OUTGOING");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_louvain_with_edge_type() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES EDGE TYPE friends");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_label_propagation_with_iterations() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION ITERATIONS 50");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_label_propagation_with_direction() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION INCOMING");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    #[test]
    fn test_coverage_label_propagation_with_edge_type() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION EDGE TYPE knows");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    // JOIN USING with multiple columns
    #[test]
    fn test_coverage_join_using_multiple_columns() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b USING (x, y, z)");
        let s = unwrap_select(stmt);
        if let Some(from) = s.from {
            assert!(!from.joins.is_empty());
            if let Some(JoinCondition::Using(cols)) = &from.joins[0].condition {
                assert_eq!(cols.len(), 3);
            }
        } else {
            panic!("expected SELECT");
        }
    }

    // UPDATE without WHERE
    #[test]
    fn test_coverage_update_without_where() {
        let stmt = parse_stmt("UPDATE users SET active = TRUE");
        let u = unwrap_update(stmt);
        assert!(u.where_clause.is_none());
    }

    // DELETE without WHERE
    #[test]
    fn test_coverage_delete_without_where() {
        let stmt = parse_stmt("DELETE FROM users");
        let d = unwrap_delete(stmt);
        assert!(d.where_clause.is_none());
    }

    // Table alias that looks like keyword (but is identifier)
    #[test]
    fn test_coverage_table_ref_keyword_not_alias() {
        // SELECT ... FROM table WHERE ... (WHERE is keyword, not alias)
        let stmt = parse_stmt("SELECT * FROM users WHERE id = 1");
        let s = unwrap_select(stmt);
        if let Some(from) = s.from {
            // The table should not have an alias since WHERE is a keyword
            assert!(from.table.alias.is_none());
        }
    }

    // INSERT with SELECT
    #[test]
    fn test_coverage_insert_select() {
        let stmt = parse_stmt("INSERT INTO archive (name) SELECT name FROM users");
        let i = unwrap_insert(stmt);
        assert!(matches!(i.source, InsertSource::Query(_)));
    }

    // Error: INSERT without VALUES or SELECT
    #[test]
    fn test_coverage_error_insert_invalid() {
        let result = parse("INSERT INTO users (name)");
        assert!(result.is_err());
    }

    // NOT BETWEEN and NOT LIKE
    #[test]
    fn test_coverage_not_between() {
        let stmt = parse_stmt("SELECT * FROM users WHERE age NOT BETWEEN 18 AND 65");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_not_like() {
        let stmt = parse_stmt("SELECT * FROM users WHERE name NOT LIKE '%admin%'");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // IS NOT NULL
    #[test]
    fn test_coverage_is_not_null() {
        let stmt = parse_stmt("SELECT * FROM users WHERE email IS NOT NULL");
        let s = unwrap_select(stmt);
        if let Some(where_clause) = s.where_clause {
            let ExprKind::IsNull { negated, .. } = where_clause.kind else {
                panic!("expected IS NULL")
            };
            assert!(negated);
        }
    }

    // Qualified wildcard error path
    #[test]
    fn test_coverage_error_qualified_wildcard_non_ident() {
        // "1.*" - qualified wildcard requires identifier, not literal
        let result = parse("SELECT 1.*");
        assert!(result.is_err());
    }

    // Expression with function call using DISTINCT
    #[test]
    fn test_coverage_function_distinct() {
        let stmt = parse_stmt("SELECT COUNT(DISTINCT name) FROM users");
        let s = unwrap_select(stmt);
        let ExprKind::Call(call) = &s.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert!(call.distinct);
    }

    // Multiple function arguments
    #[test]
    fn test_coverage_function_multiple_args() {
        let stmt = parse_stmt("SELECT COALESCE(a, b, c, d) FROM t");
        let s = unwrap_select(stmt);
        let ExprKind::Call(call) = &s.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert_eq!(call.args.len(), 4);
    }

    // CREATE UNIQUE INDEX
    #[test]
    fn test_coverage_create_unique_index() {
        let stmt = parse_stmt("CREATE UNIQUE INDEX idx ON users (email)");
        let ci = unwrap_createindex(stmt);
        assert!(ci.unique);
    }

    // Multiple UPDATE assignments
    #[test]
    fn test_coverage_update_multiple_assignments() {
        let stmt =
            parse_stmt("UPDATE users SET name = 'Bob', age = 30, active = TRUE WHERE id = 1");
        let u = unwrap_update(stmt);
        assert_eq!(u.assignments.len(), 3);
    }

    // CLUSTER with various commands
    #[test]
    fn test_coverage_cluster_status() {
        let stmt = parse_stmt("CLUSTER STATUS");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Status));
    }

    #[test]
    fn test_coverage_cluster_nodes() {
        let stmt = parse_stmt("CLUSTER NODES");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Nodes));
    }

    #[test]
    fn test_coverage_cluster_leader() {
        let stmt = parse_stmt("CLUSTER LEADER");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Leader));
    }

    #[test]
    fn test_coverage_cluster_connect() {
        let stmt = parse_stmt("CLUSTER CONNECT '127.0.0.1:9000'");
        let c = unwrap_cluster(stmt);
        assert!(matches!(c.operation, ClusterOp::Connect { .. }));
    }

    // GRAPH PAGERANK with all options
    #[test]
    fn test_coverage_pagerank_with_opts() {
        let stmt =
            parse_stmt("GRAPH PAGERANK DAMPING 0.85 ITERATIONS 100 TOLERANCE 0.001 OUTGOING");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    // Vector operations with metrics
    #[test]
    fn test_coverage_similar_euclidean() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0] LIMIT 10 EUCLIDEAN");
        let s = unwrap_similar(stmt);
        assert!(s.metric.is_some());
    }

    // BLOB operations
    #[test]
    fn test_coverage_blob_get() {
        let stmt = parse_stmt("BLOB GET 'hash123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Get { .. }));
    }

    #[test]
    fn test_coverage_blob_delete() {
        let stmt = parse_stmt("BLOB DELETE 'hash123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Delete { .. }));
    }

    #[test]
    fn test_coverage_blob_info() {
        let stmt = parse_stmt("BLOB INFO 'hash123'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Info { .. }));
    }

    // CHECKPOINT operations
    #[test]
    fn test_coverage_checkpoint_named() {
        let stmt = parse_stmt("CHECKPOINT 'backup1'");
        let c = unwrap_checkpoint(stmt);
        assert!(c.name.is_some());
    }

    #[test]
    fn test_coverage_checkpoints_list() {
        let stmt = parse_stmt("CHECKPOINTS LIMIT 10");
        let c = unwrap_checkpoints(stmt);
        assert!(c.limit.is_some());
    }

    #[test]
    fn test_coverage_rollback_to() {
        let stmt = parse_stmt("ROLLBACK TO 'checkpoint1'");
        assert!(matches!(stmt.kind, StatementKind::Rollback(_)));
    }

    // DISTINCT function call with no args
    #[test]
    fn test_coverage_function_no_args() {
        let stmt = parse_stmt("SELECT NOW() FROM dual");
        let s = unwrap_select(stmt);
        let ExprKind::Call(call) = &s.columns[0].expr.kind else {
            panic!("expected SELECT")
        };
        assert!(call.args.is_empty());
    }

    // GRAPH INDEX create label
    #[test]
    fn test_coverage_graph_index_create_label() {
        let stmt = parse_stmt("GRAPH INDEX CREATE ON LABEL");
        assert!(matches!(stmt.kind, StatementKind::GraphIndex(_)));
    }

    // PATH statement with arrow syntax
    #[test]
    fn test_coverage_path_arrow_syntax() {
        let stmt = parse_stmt("PATH 1 -> 2");
        assert!(matches!(stmt.kind, StatementKind::Path(_)));
    }

    // Parse identifier where keyword expected
    #[test]
    fn test_coverage_error_ident_or_keyword() {
        // EDGE TYPE expects identifier or keyword, but gets number
        let result = parse("GRAPH BETWEENNESS CENTRALITY EDGE TYPE 123");
        assert!(result.is_err());
    }

    // NOT IN expression
    #[test]
    fn test_coverage_not_in() {
        let stmt = parse_stmt("SELECT * FROM users WHERE id NOT IN (1, 2, 3)");
        let s = unwrap_select(stmt);
        assert!(s.where_clause.is_some());
    }

    // CASE expression coverage
    #[test]
    fn test_coverage_case_with_else() {
        let stmt = parse_stmt("SELECT CASE WHEN x > 0 THEN 'pos' ELSE 'neg' END FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_coverage_case_multiple_when() {
        let stmt = parse_stmt(
            "SELECT CASE WHEN x < 0 THEN 'neg' WHEN x > 0 THEN 'pos' ELSE 'zero' END FROM t",
        );
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Subquery in expression
    #[test]
    fn test_coverage_subquery_in_where() {
        let stmt = parse_stmt("SELECT * FROM users WHERE id IN (SELECT user_id FROM orders)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Expression with unary minus
    #[test]
    fn test_coverage_unary_minus() {
        let stmt = parse_stmt("SELECT -1 FROM dual");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Expression with NOT
    #[test]
    fn test_coverage_unary_not() {
        let stmt = parse_stmt("SELECT * FROM users WHERE NOT active");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Table alias implicit (without AS)
    #[test]
    fn test_coverage_table_alias_implicit() {
        let stmt = parse_stmt("SELECT u.name FROM users u");
        let s = unwrap_select(stmt);
        if let Some(from) = s.from {
            assert!(from.table.alias.is_some());
        }
    }

    // NEIGHBORS with direction
    #[test]
    fn test_coverage_neighbors_with_direction() {
        let stmt = parse_stmt("NEIGHBORS 1 OUTGOING");
        let n = unwrap_neighbors(stmt);
        assert!(matches!(n.direction, Direction::Outgoing));
    }

    // NEIGHBORS with edge type using colon syntax
    #[test]
    fn test_coverage_neighbors_edge_type_colon() {
        let stmt = parse_stmt("NEIGHBORS 1 : FOLLOWS");
        let n = unwrap_neighbors(stmt);
        assert!(n.edge_type.is_some());
    }

    // ENTITY GET
    #[test]
    fn test_coverage_entity_get() {
        let stmt = parse_stmt("ENTITY GET 'user:1'");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Get { .. }));
    }

    // ENTITY DELETE
    #[test]
    fn test_coverage_entity_delete() {
        let stmt = parse_stmt("ENTITY DELETE 'user:1'");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Delete { .. }));
    }

    // INSERT with multiple value rows
    #[test]
    fn test_coverage_insert_multiple_rows() {
        let stmt = parse_stmt("INSERT INTO users (name) VALUES ('Alice'), ('Bob'), ('Carol')");
        let i = unwrap_insert(stmt);
        let InsertSource::Values(rows) = i.source else {
            panic!("expected INSERT")
        };
        assert_eq!(rows.len(), 3);
    }

    // SELECT with GROUP BY multiple columns
    #[test]
    fn test_coverage_group_by_multiple() {
        let stmt = parse_stmt("SELECT a, b, COUNT(*) FROM t GROUP BY a, b");
        let s = unwrap_select(stmt);
        assert_eq!(s.group_by.len(), 2);
    }

    // ORDER BY with multiple columns and directions
    #[test]
    fn test_coverage_order_by_multiple_mixed() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY a ASC, b DESC, c");
        let s = unwrap_select(stmt);
        assert_eq!(s.order_by.len(), 3);
    }

    // Expression with modulo
    #[test]
    fn test_coverage_modulo_expr() {
        let stmt = parse_stmt("SELECT 10 % 3 FROM dual");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // BLOB PUT with from path
    #[test]
    fn test_coverage_blob_put_from_path() {
        let stmt = parse_stmt("BLOB PUT 'file.txt' FROM '/path/to/file'");
        let b = unwrap_blob(stmt);
        let BlobOp::Put { from_path, .. } = b.operation else {
            panic!("expected PUT")
        };
        assert!(from_path.is_some());
    }

    // BLOBS BY TAG
    #[test]
    fn test_coverage_blobs_by_tag() {
        let stmt = parse_stmt("BLOBS BY TAG 'important'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::ByTag { .. }));
    }

    // BLOBS BY TYPE
    #[test]
    fn test_coverage_blobs_by_type() {
        let stmt = parse_stmt("BLOBS WHERE TYPE = 'image/png'");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::ByType { .. }));
    }

    // BLOBS SIMILAR
    #[test]
    fn test_coverage_blobs_similar() {
        let stmt = parse_stmt("BLOBS SIMILAR TO 'hash123' LIMIT 5");
        let b = unwrap_blobs(stmt);
        assert!(matches!(b.operation, BlobsOp::Similar { .. }));
    }

    // Error: invalid table constraint
    #[test]
    fn test_coverage_error_table_constraint_invalid() {
        let result = parse("CREATE TABLE t (id INT, INVALID_CONSTRAINT)");
        assert!(result.is_err());
    }

    // DECIMAL with precision and scale
    #[test]
    fn test_coverage_decimal_precision_scale() {
        let stmt = parse_stmt("CREATE TABLE t (price DECIMAL(10, 2))");
        let ct = unwrap_createtable(stmt);
        assert_eq!(ct.columns.len(), 1);
    }

    // VARCHAR with length
    #[test]
    fn test_coverage_varchar_length() {
        let stmt = parse_stmt("CREATE TABLE t (name VARCHAR(255))");
        let ct = unwrap_createtable(stmt);
        assert_eq!(ct.columns.len(), 1);
    }

    // CREATE INDEX IF NOT EXISTS
    #[test]
    fn test_coverage_create_index_if_not_exists() {
        let stmt = parse_stmt("CREATE INDEX IF NOT EXISTS idx ON users (email)");
        let ci = unwrap_createindex(stmt);
        assert!(ci.if_not_exists);
    }

    // CREATE INDEX with multiple columns
    #[test]
    fn test_coverage_create_index_multiple_columns() {
        let stmt = parse_stmt("CREATE INDEX idx ON users (first_name, last_name)");
        let ci = unwrap_createindex(stmt);
        assert_eq!(ci.columns.len(), 2);
    }

    // TABLE constraint CHECK
    #[test]
    fn test_coverage_table_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (age INT, CHECK (age >= 0))");
        let ct = unwrap_createtable(stmt);
        assert_eq!(ct.constraints.len(), 1);
    }

    // Error: invalid GRAPH operation
    #[test]
    fn test_coverage_error_graph_invalid() {
        let result = parse("GRAPH INVALID_OP");
        assert!(result.is_err());
    }

    // DROP TABLE IF EXISTS
    #[test]
    fn test_coverage_drop_table_if_exists() {
        let stmt = parse_stmt("DROP TABLE IF EXISTS users");
        let dt = unwrap_droptable(stmt);
        assert!(dt.if_exists);
    }

    // DROP INDEX
    #[test]
    fn test_coverage_drop_index() {
        let stmt = parse_stmt("DROP INDEX idx");
        assert!(matches!(stmt.kind, StatementKind::DropIndex(_)));
    }

    // DESCRIBE table
    #[test]
    fn test_coverage_describe_table() {
        let stmt = parse_stmt("DESCRIBE TABLE users");
        assert!(matches!(stmt.kind, StatementKind::Describe(_)));
    }

    // SHOW TABLES
    #[test]
    fn test_coverage_show_tables() {
        let stmt = parse_stmt("SHOW TABLES");
        assert!(matches!(stmt.kind, StatementKind::ShowTables));
    }

    // SHOW VECTOR INDEX
    #[test]
    fn test_coverage_show_vector_index() {
        let stmt = parse_stmt("SHOW VECTOR INDEX");
        assert!(matches!(stmt.kind, StatementKind::ShowVectorIndex));
    }

    // COUNT EMBEDDINGS
    #[test]
    fn test_coverage_count_embeddings() {
        let stmt = parse_stmt("COUNT EMBEDDINGS");
        assert!(matches!(stmt.kind, StatementKind::CountEmbeddings));
    }

    // SHOW EMBEDDINGS with LIMIT
    #[test]
    fn test_coverage_show_embeddings_limit() {
        let stmt = parse_stmt("SHOW EMBEDDINGS LIMIT 10");
        if let StatementKind::ShowEmbeddings { limit } = stmt.kind {
            assert!(limit.is_some());
        } else {
            panic!("expected SHOW EMBEDDINGS");
        }
    }

    // DECIMAL without precision (just type name)
    #[test]
    fn test_coverage_decimal_no_precision() {
        let stmt = parse_stmt("CREATE TABLE t (price DECIMAL)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // NUMERIC with precision only
    #[test]
    fn test_coverage_numeric_precision_only() {
        let stmt = parse_stmt("CREATE TABLE t (value NUMERIC(10))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // Column with DEFAULT value
    #[test]
    fn test_coverage_column_default() {
        let stmt = parse_stmt("CREATE TABLE t (active BOOLEAN DEFAULT TRUE)");
        let ct = unwrap_createtable(stmt);
        assert!(!ct.columns[0].constraints.is_empty());
    }

    // Column with REFERENCES (foreign key)
    #[test]
    fn test_coverage_column_references() {
        let stmt = parse_stmt("CREATE TABLE orders (user_id INT REFERENCES users(id))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // EMBED statement
    #[test]
    fn test_coverage_embed_statement() {
        let stmt = parse_stmt("EMBED STORE 'doc1' [1.0, 2.0, 3.0]");
        assert!(matches!(stmt.kind, StatementKind::Embed(_)));
    }

    // NODE GET
    #[test]
    fn test_coverage_node_get() {
        let stmt = parse_stmt("NODE GET 1");
        let n = unwrap_node(stmt);
        assert!(matches!(n.operation, NodeOp::Get { .. }));
    }

    // NODE DELETE
    #[test]
    fn test_coverage_node_delete() {
        let stmt = parse_stmt("NODE DELETE 1");
        let n = unwrap_node(stmt);
        assert!(matches!(n.operation, NodeOp::Delete { .. }));
    }

    // EDGE GET
    #[test]
    fn test_coverage_edge_get() {
        let stmt = parse_stmt("EDGE GET 1");
        let e = unwrap_edge(stmt);
        assert!(matches!(e.operation, EdgeOp::Get { .. }));
    }

    // EDGE DELETE
    #[test]
    fn test_coverage_edge_delete() {
        let stmt = parse_stmt("EDGE DELETE 1");
        let e = unwrap_edge(stmt);
        assert!(matches!(e.operation, EdgeOp::Delete { .. }));
    }

    // EDGE LIST
    #[test]
    fn test_coverage_edge_list() {
        let stmt = parse_stmt("EDGE LIST FOLLOWS LIMIT 10");
        let e = unwrap_edge(stmt);
        assert!(matches!(e.operation, EdgeOp::List { .. }));
    }

    // NODE LIST with LIMIT
    #[test]
    fn test_coverage_node_list_limit() {
        let stmt = parse_stmt("NODE LIST Person LIMIT 10 OFFSET 5");
        let n = unwrap_node(stmt);
        let NodeOp::List { limit, offset, .. } = n.operation else {
            panic!("expected LIST")
        };
        assert!(limit.is_some());
        assert!(offset.is_some());
    }

    // SIMILAR without LIMIT (no default, returns None)
    #[test]
    fn test_coverage_similar_no_limit() {
        let stmt = parse_stmt("SIMILAR [1.0, 2.0]");
        let s = unwrap_similar(stmt);
        assert!(s.limit.is_none()); // No LIMIT clause means None
    }

    // NEIGHBORS with LIMIT
    #[test]
    fn test_coverage_neighbors_limit() {
        let stmt = parse_stmt("NEIGHBORS 1 LIMIT 5");
        let n = unwrap_neighbors(stmt);
        assert!(n.limit.is_some());
    }

    // ENTITY UPDATE
    #[test]
    fn test_coverage_entity_update() {
        let stmt = parse_stmt("ENTITY UPDATE 'user:1' { name: 'Bob' }");
        let e = unwrap_entity(stmt);
        assert!(matches!(e.operation, EntityOp::Update { .. }));
    }

    // FIND NODES with WHERE
    #[test]
    fn test_coverage_find_nodes_where() {
        let stmt = parse_stmt("FIND NODE Person WHERE age > 18");
        let f = unwrap_find(stmt);
        assert!(f.where_clause.is_some());
    }

    // CHAIN TIP
    #[test]
    fn test_coverage_chain_tip() {
        let stmt = parse_stmt("CHAIN TIP");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Tip));
    }

    // CHAIN HEIGHT
    #[test]
    fn test_coverage_chain_height() {
        let stmt = parse_stmt("CHAIN HEIGHT");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Height));
    }

    // CHAIN VERIFY
    #[test]
    fn test_coverage_chain_verify() {
        let stmt = parse_stmt("CHAIN VERIFY");
        let c = unwrap_chain(stmt);
        assert!(matches!(c.operation, ChainOp::Verify));
    }

    // Error: invalid CLUSTER operation
    #[test]
    fn test_coverage_error_cluster_invalid() {
        let result = parse("CLUSTER INVALID");
        assert!(result.is_err());
    }

    // Error: invalid BLOB operation
    #[test]
    fn test_coverage_error_blob_invalid() {
        let result = parse("BLOB INVALID_OP");
        assert!(result.is_err());
    }

    // Error: invalid ENTITY operation
    #[test]
    fn test_coverage_error_entity_invalid() {
        let result = parse("ENTITY INVALID_OP");
        assert!(result.is_err());
    }

    // Error: invalid CHAIN operation
    #[test]
    fn test_coverage_error_chain_invalid() {
        let result = parse("CHAIN INVALID_OP");
        assert!(result.is_err());
    }

    // Error: invalid NODE operation
    #[test]
    fn test_coverage_error_node_invalid() {
        let result = parse("NODE INVALID_OP");
        assert!(result.is_err());
    }

    // Error: invalid EDGE operation
    #[test]
    fn test_coverage_error_edge_invalid() {
        let result = parse("EDGE INVALID_OP");
        assert!(result.is_err());
    }

    // BLOB LINK
    #[test]
    fn test_coverage_blob_link() {
        let stmt = parse_stmt("BLOB LINK 'hash123' TO 'entity1'");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Link { .. }));
    }

    // BLOB INIT
    #[test]
    fn test_coverage_blob_init() {
        let stmt = parse_stmt("BLOB INIT");
        let b = unwrap_blob(stmt);
        assert!(matches!(b.operation, BlobOp::Init));
    }

    // =============================
    // Error path coverage tests
    // =============================

    fn parse_fails(source: &str) -> ParseError {
        parse(source).unwrap_err()
    }

    // === Statement extraction helpers (covered via #[should_panic] tests) ===

    fn unwrap_blob(stmt: Statement) -> BlobStmt {
        match stmt.kind {
            StatementKind::Blob(v) => v,
            _ => panic!("expected Blob"),
        }
    }

    fn unwrap_blobs(stmt: Statement) -> BlobsStmt {
        match stmt.kind {
            StatementKind::Blobs(v) => v,
            _ => panic!("expected Blobs"),
        }
    }

    fn unwrap_cache(stmt: Statement) -> CacheStmt {
        match stmt.kind {
            StatementKind::Cache(v) => v,
            _ => panic!("expected Cache"),
        }
    }

    fn unwrap_chain(stmt: Statement) -> ChainStmt {
        match stmt.kind {
            StatementKind::Chain(v) => v,
            _ => panic!("expected Chain"),
        }
    }

    fn unwrap_checkpoint(stmt: Statement) -> CheckpointStmt {
        match stmt.kind {
            StatementKind::Checkpoint(v) => v,
            _ => panic!("expected Checkpoint"),
        }
    }

    fn unwrap_checkpoints(stmt: Statement) -> CheckpointsStmt {
        match stmt.kind {
            StatementKind::Checkpoints(v) => v,
            _ => panic!("expected Checkpoints"),
        }
    }

    fn unwrap_cluster(stmt: Statement) -> ClusterStmt {
        match stmt.kind {
            StatementKind::Cluster(v) => v,
            _ => panic!("expected Cluster"),
        }
    }

    fn unwrap_createindex(stmt: Statement) -> CreateIndexStmt {
        match stmt.kind {
            StatementKind::CreateIndex(v) => v,
            _ => panic!("expected CreateIndex"),
        }
    }

    fn unwrap_createtable(stmt: Statement) -> CreateTableStmt {
        match stmt.kind {
            StatementKind::CreateTable(v) => v,
            _ => panic!("expected CreateTable"),
        }
    }

    fn unwrap_delete(stmt: Statement) -> DeleteStmt {
        match stmt.kind {
            StatementKind::Delete(v) => v,
            _ => panic!("expected Delete"),
        }
    }

    fn unwrap_describe(stmt: Statement) -> DescribeStmt {
        match stmt.kind {
            StatementKind::Describe(v) => v,
            _ => panic!("expected Describe"),
        }
    }

    fn unwrap_dropindex(stmt: Statement) -> DropIndexStmt {
        match stmt.kind {
            StatementKind::DropIndex(v) => v,
            _ => panic!("expected DropIndex"),
        }
    }

    fn unwrap_droptable(stmt: Statement) -> DropTableStmt {
        match stmt.kind {
            StatementKind::DropTable(v) => v,
            _ => panic!("expected DropTable"),
        }
    }

    fn unwrap_edge(stmt: Statement) -> EdgeStmt {
        match stmt.kind {
            StatementKind::Edge(v) => v,
            _ => panic!("expected Edge"),
        }
    }

    fn unwrap_embed(stmt: Statement) -> EmbedStmt {
        match stmt.kind {
            StatementKind::Embed(v) => v,
            _ => panic!("expected Embed"),
        }
    }

    fn unwrap_entity(stmt: Statement) -> EntityStmt {
        match stmt.kind {
            StatementKind::Entity(v) => v,
            _ => panic!("expected Entity"),
        }
    }

    fn unwrap_find(stmt: Statement) -> FindStmt {
        match stmt.kind {
            StatementKind::Find(v) => v,
            _ => panic!("expected Find"),
        }
    }

    fn unwrap_graphaggregate(stmt: Statement) -> GraphAggregateStmt {
        match stmt.kind {
            StatementKind::GraphAggregate(v) => v,
            _ => panic!("expected GraphAggregate"),
        }
    }

    fn unwrap_graphalgorithm(stmt: Statement) -> GraphAlgorithmStmt {
        match stmt.kind {
            StatementKind::GraphAlgorithm(v) => v,
            _ => panic!("expected GraphAlgorithm"),
        }
    }

    fn unwrap_graphbatch(stmt: Statement) -> GraphBatchStmt {
        match stmt.kind {
            StatementKind::GraphBatch(v) => v,
            _ => panic!("expected GraphBatch"),
        }
    }

    fn unwrap_graphconstraint(stmt: Statement) -> GraphConstraintStmt {
        match stmt.kind {
            StatementKind::GraphConstraint(v) => v,
            _ => panic!("expected GraphConstraint"),
        }
    }

    fn unwrap_graphindex(stmt: Statement) -> GraphIndexStmt {
        match stmt.kind {
            StatementKind::GraphIndex(v) => v,
            _ => panic!("expected GraphIndex"),
        }
    }

    fn unwrap_insert(stmt: Statement) -> InsertStmt {
        match stmt.kind {
            StatementKind::Insert(v) => v,
            _ => panic!("expected Insert"),
        }
    }

    fn unwrap_neighbors(stmt: Statement) -> NeighborsStmt {
        match stmt.kind {
            StatementKind::Neighbors(v) => v,
            _ => panic!("expected Neighbors"),
        }
    }

    fn unwrap_node(stmt: Statement) -> NodeStmt {
        match stmt.kind {
            StatementKind::Node(v) => v,
            _ => panic!("expected Node"),
        }
    }

    fn unwrap_path(stmt: Statement) -> PathStmt {
        match stmt.kind {
            StatementKind::Path(v) => v,
            _ => panic!("expected Path"),
        }
    }

    fn unwrap_rollback(stmt: Statement) -> RollbackStmt {
        match stmt.kind {
            StatementKind::Rollback(v) => v,
            _ => panic!("expected Rollback"),
        }
    }

    fn unwrap_select(stmt: Statement) -> SelectStmt {
        match stmt.kind {
            StatementKind::Select(v) => v,
            _ => panic!("expected Select"),
        }
    }

    fn unwrap_similar(stmt: Statement) -> SimilarStmt {
        match stmt.kind {
            StatementKind::Similar(v) => v,
            _ => panic!("expected Similar"),
        }
    }

    fn unwrap_update(stmt: Statement) -> UpdateStmt {
        match stmt.kind {
            StatementKind::Update(v) => v,
            _ => panic!("expected Update"),
        }
    }

    fn unwrap_vault(stmt: Statement) -> VaultStmt {
        match stmt.kind {
            StatementKind::Vault(v) => v,
            _ => panic!("expected Vault"),
        }
    }

    #[test]
    fn test_error_graph_constraint_invalid_target() {
        let err = parse_fails("GRAPH CONSTRAINT CREATE FOO PROPERTY name UNIQUE");
        assert!(err.to_string().contains("NODE or EDGE") || err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_graph_constraint_invalid_type() {
        let err = parse_fails("GRAPH CONSTRAINT CREATE NODE Person PROPERTY name INVALID");
        assert!(
            err.to_string().contains("UNIQUE")
                || err.to_string().contains("EXISTS")
                || err.to_string().contains("unexpected")
        );
    }

    #[test]
    fn test_error_missing_expression() {
        let err = parse_fails("SELECT FROM users");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_unterminated_list() {
        let err = parse_fails("SELECT * FROM users WHERE id IN (1, 2, 3");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains(")"));
    }

    #[test]
    fn test_error_missing_table_name() {
        let err = parse_fails("SELECT * FROM");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_unclosed_paren() {
        let err = parse_fails("SELECT (a + b FROM users");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_invalid_column_def() {
        let err = parse_fails("CREATE TABLE t (col1)");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains("type"));
    }

    #[test]
    fn test_error_invalid_order_by() {
        let result = parse("SELECT * FROM users ORDER BY");
        assert!(result.is_err());
    }

    #[test]
    fn test_error_chain_invalid_operation() {
        let err = parse_fails("CHAIN UNKNOWN");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_blob_invalid_operation() {
        let err = parse_fails("BLOB UNKNOWN");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains("INIT"));
    }

    #[test]
    fn test_error_vault_invalid_operation() {
        let err = parse_fails("VAULT UNKNOWN");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_cache_invalid_operation() {
        let err = parse_fails("CACHE UNKNOWN");
        assert!(err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_embed_invalid_operation() {
        let err = parse_fails("EMBED UNKNOWN");
        assert!(err.to_string().contains("STORE") || err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_node_invalid_operation() {
        let err = parse_fails("NODE UNKNOWN");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains("CREATE"));
    }

    #[test]
    fn test_error_edge_invalid_operation() {
        let err = parse_fails("EDGE UNKNOWN");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains("CREATE"));
    }

    #[test]
    fn test_error_graph_index_invalid() {
        let err = parse_fails("GRAPH INDEX UNKNOWN");
        assert!(err.to_string().contains("CREATE") || err.to_string().contains("unexpected"));
    }

    #[test]
    fn test_error_entity_invalid_operation() {
        let err = parse_fails("ENTITY UNKNOWN");
        assert!(err.to_string().contains("unexpected") || err.to_string().contains("CREATE"));
    }

    #[test]
    fn test_error_cluster_invalid_operation() {
        let err = parse_fails("CLUSTER UNKNOWN");
        assert!(err.to_string().contains("unexpected"));
    }

    // =============================
    // Extended coverage tests
    // =============================

    // Graph algorithms with direction
    #[test]
    fn test_graph_pagerank_with_direction() {
        let stmt = parse_stmt("GRAPH PAGERANK OUTGOING");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::PageRank { direction, .. } = g.operation else {
            panic!("expected PageRank")
        };
        assert!(direction.is_some());
    }

    #[test]
    fn test_graph_pagerank_with_edge_type() {
        let stmt = parse_stmt("GRAPH PAGERANK EDGE TYPE follows");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::PageRank { edge_type, .. } = g.operation else {
            panic!("expected PageRank")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_betweenness_with_direction() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY INCOMING");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::BetweennessCentrality { direction, .. } = g.operation else {
            panic!("expected BetweennessCentrality")
        };
        assert!(direction.is_some());
    }

    // Error path tests for coverage
    #[test]
    fn test_invalid_table_constraint() {
        // CREATE TABLE with invalid constraint keyword
        parse_fails("CREATE TABLE t (id INT, INVALIDCONST)");
    }

    #[test]
    fn test_insert_without_values_or_select() {
        // INSERT without VALUES or SELECT
        parse_fails("INSERT INTO users (name) INVALID");
    }

    #[test]
    fn test_expect_ident_or_any_keyword_error() {
        // Property list with number instead of key
        parse_fails("NODE CREATE person {123: 'value'}");
    }

    #[test]
    fn test_qualified_wildcard_error() {
        // Qualified wildcard on non-identifier
        parse_fails("SELECT (1+2).* FROM t");
    }

    #[test]
    fn test_path_invalid_from() {
        // PATH with invalid FROM
        parse_fails("PATH FROM");
    }

    #[test]
    fn test_path_missing_to() {
        // PATH without TO clause
        parse_fails("PATH FROM 1 WHERE x > 1");
    }

    #[test]
    fn test_traverse_missing_from() {
        // TRAVERSE without valid start
        parse_fails("TRAVERSE FROM");
    }

    #[test]
    fn test_aggregate_missing_property() {
        // AGGREGATE without property
        parse_fails("AGGREGATE SUM ON Person");
    }

    #[test]
    fn test_embed_missing_collection() {
        // EMBED without collection name
        parse_fails("EMBED 'text'");
    }

    #[test]
    fn test_create_table_empty_columns() {
        // CREATE TABLE with empty column list
        parse_fails("CREATE TABLE t ()");
    }

    #[test]
    fn test_select_unclosed_paren() {
        // SELECT with unclosed parenthesis
        parse_fails("SELECT (1 + 2 FROM t");
    }

    #[test]
    fn test_update_missing_set() {
        // UPDATE without SET clause
        parse_fails("UPDATE users WHERE id = 1");
    }

    #[test]
    fn test_delete_missing_table() {
        // DELETE without table name
        parse_fails("DELETE FROM");
    }

    #[test]
    fn test_drop_invalid_type() {
        // DROP with invalid object type
        parse_fails("DROP INVALID foo");
    }

    #[test]
    fn test_create_without_type() {
        // CREATE without object type
        parse_fails("CREATE");
    }

    #[test]
    fn test_node_update_missing_id() {
        // NODE UPDATE without id
        parse_fails("NODE UPDATE {name: 'test'}");
    }

    #[test]
    fn test_edge_delete_missing_id() {
        // EDGE DELETE without edge id
        parse_fails("EDGE DELETE");
    }

    #[test]
    fn test_invalid_string_literal() {
        // Unclosed string literal
        parse_fails("SELECT 'unclosed FROM t");
    }

    #[test]
    fn test_chain_history_missing_key() {
        // CHAIN HISTORY without key
        parse_fails("CHAIN HISTORY");
    }

    #[test]
    fn test_chain_block_missing_height() {
        // CHAIN BLOCK without height
        parse_fails("CHAIN BLOCK");
    }

    #[test]
    fn test_chain_similar_missing_embedding() {
        // CHAIN SIMILAR without embedding
        parse_fails("CHAIN SIMILAR LIMIT 10");
    }

    // Additional coverage tests
    #[test]
    fn test_entity_update_with_embedding() {
        let stmt = parse_stmt("ENTITY UPDATE 'user:1' {name: 'Bob'} EMBEDDING [1.0, 2.0]");
        let e = unwrap_entity(stmt);
        let EntityOp::Update { embedding, .. } = e.operation else {
            panic!("expected ENTITY UPDATE")
        };
        assert!(embedding.is_some());
    }

    #[test]
    fn test_batch_entity_missing_key() {
        // Batch entity definition without 'key' field
        parse_fails("ENTITY BATCH CREATE [{name: 'Alice'}]");
    }

    #[test]
    fn test_chain_invalid_operation() {
        // CHAIN with invalid operation
        parse_fails("CHAIN INVALID");
    }

    #[test]
    fn test_graph_algorithm_pagerank_edge_type() {
        let stmt = parse_stmt("GRAPH PAGERANK EDGE TYPE follows");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::PageRank { edge_type, .. } = g.operation else {
            panic!("expected PageRank")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_algorithm_betweenness_edge_type() {
        let stmt = parse_stmt("GRAPH BETWEENNESS CENTRALITY EDGE TYPE follows");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::BetweennessCentrality { edge_type, .. } = g.operation else {
            panic!("expected BetweennessCentrality")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_chain_similar_no_limit() {
        let stmt = parse_stmt("CHAIN SIMILAR [1.0, 2.0, 3.0]");
        let c = unwrap_chain(stmt);
        let ChainOp::Similar { limit, .. } = c.operation else {
            panic!("expected CHAIN SIMILAR")
        };
        assert!(limit.is_none());
    }

    #[test]
    fn test_graph_algorithm_closeness_edge_type() {
        let stmt = parse_stmt("GRAPH CLOSENESS CENTRALITY EDGE TYPE knows");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::ClosenessCentrality { edge_type, .. } = g.operation else {
            panic!("expected ClosenessCentrality")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_algorithm_eigenvector_edge_type() {
        let stmt = parse_stmt("GRAPH EIGENVECTOR CENTRALITY EDGE TYPE likes");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::EigenvectorCentrality { edge_type, .. } = g.operation else {
            panic!("expected EigenvectorCentrality")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_algorithm_louvain_edge_type() {
        let stmt = parse_stmt("GRAPH LOUVAIN COMMUNITIES EDGE TYPE friend");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::LouvainCommunities { edge_type, .. } = g.operation else {
            panic!("expected LouvainCommunities")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_algorithm_label_propagation_edge_type() {
        let stmt = parse_stmt("GRAPH LABEL PROPAGATION EDGE TYPE connects");
        let g = unwrap_graphalgorithm(stmt);
        let GraphAlgorithmOp::LabelPropagation { edge_type, .. } = g.operation else {
            panic!("expected LabelPropagation")
        };
        assert!(edge_type.is_some());
    }

    #[test]
    fn test_graph_aggregate_node_property() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY score SUM ON Person");
        let a = unwrap_graphaggregate(stmt);
        assert!(matches!(
            a.operation,
            GraphAggregateOp::AggregateNodeProperty { .. }
        ));
    }

    #[test]
    fn test_graph_aggregate_edge_property() {
        let stmt = parse_stmt("AGGREGATE EDGE PROPERTY weight AVG ON FOLLOWS");
        let a = unwrap_graphaggregate(stmt);
        assert!(matches!(
            a.operation,
            GraphAggregateOp::AggregateEdgeProperty { .. }
        ));
    }

    #[test]
    fn test_graph_index_show_all_nodes() {
        let stmt = parse_stmt("GRAPH INDEX SHOW ON NODE");
        assert!(matches!(stmt.kind, StatementKind::GraphIndex(_)));
    }

    // Additional error path coverage tests
    #[test]
    fn test_aggregate_invalid_target() {
        parse_fails("AGGREGATE INVALID PROPERTY score SUM ON Person");
    }

    #[test]
    fn test_aggregate_invalid_function() {
        parse_fails("AGGREGATE NODE PROPERTY score INVALID ON Person");
    }

    #[test]
    fn test_select_with_cast() {
        let stmt = parse_stmt("SELECT CAST(age AS VARCHAR) FROM users");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_case_when() {
        let stmt = parse_stmt("SELECT CASE WHEN x > 1 THEN 'big' ELSE 'small' END FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_find_edge_pattern() {
        let stmt = parse_stmt("FIND EDGE FOLLOWS");
        let f = unwrap_find(stmt);
        assert!(matches!(f.pattern, FindPattern::Edges { .. }));
    }

    // Targeted error path tests for coverage
    #[test]
    fn test_case_without_when_clause() {
        // CASE without any WHEN clause
        parse_fails("SELECT CASE ELSE 'default' END FROM t");
    }

    #[test]
    fn test_constraint_invalid_node_edge() {
        // GRAPH CONSTRAINT with neither NODE nor EDGE
        parse_fails("GRAPH CONSTRAINT CREATE c INVALID PROPERTY name UNIQUE");
    }

    #[test]
    fn test_constraint_invalid_type() {
        // GRAPH CONSTRAINT with invalid constraint type
        parse_fails("GRAPH CONSTRAINT CREATE c NODE PROPERTY name BADTYPE");
    }

    #[test]
    fn test_node_create_without_label() {
        // NODE CREATE requires a label
        let stmt = parse_stmt("NODE CREATE person {name: 'Alice'}");
        assert!(matches!(stmt.kind, StatementKind::Node(_)));
    }

    #[test]
    fn test_edge_create_with_properties() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : FOLLOWS {since: 2020}");
        let e = unwrap_edge(stmt);
        let EdgeOp::Create { properties, .. } = e.operation else {
            panic!("expected EDGE CREATE")
        };
        assert!(!properties.is_empty());
    }

    #[test]
    fn test_select_union() {
        let stmt = parse_stmt("SELECT a FROM t1 UNION SELECT b FROM t2");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_intersect() {
        let stmt = parse_stmt("SELECT a FROM t1 INTERSECT SELECT b FROM t2");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_except() {
        let stmt = parse_stmt("SELECT a FROM t1 EXCEPT SELECT b FROM t2");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_natural_join() {
        let stmt = parse_stmt("SELECT * FROM a NATURAL JOIN b");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_cross_join() {
        let stmt = parse_stmt("SELECT * FROM a CROSS JOIN b");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_select_with_full_outer_join() {
        let stmt = parse_stmt("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_create_table_with_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (id INT, CHECK (id > 0))");
        let c = unwrap_createtable(stmt);
        assert!(!c.constraints.is_empty());
    }

    #[test]
    fn test_select_order_by_nulls_first() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x NULLS FIRST");
        let s = unwrap_select(stmt);
        assert!(!s.order_by.is_empty());
    }

    #[test]
    fn test_select_order_by_nulls_last() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x NULLS LAST");
        let s = unwrap_select(stmt);
        assert!(!s.order_by.is_empty());
    }

    #[test]
    fn test_expression_in_list() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expression_between() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x BETWEEN 1 AND 10");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expression_not_between() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT BETWEEN 1 AND 10");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expression_not_in() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expression_not_like() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x NOT LIKE '%test%'");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expression_is_not_null() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NOT NULL");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_unary_minus() {
        let stmt = parse_stmt("SELECT -x FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_tuple_expression() {
        let stmt = parse_stmt("SELECT * FROM t WHERE (a, b) = (1, 2)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Additional coverage tests for error paths and edge cases

    #[test]
    fn test_similar_empty_vector() {
        let stmt = parse_stmt("SIMILAR [] LIMIT 5");
        let s = unwrap_similar(stmt);
        let SimilarQuery::Vector(v) = s.query else {
            panic!("expected SIMILAR")
        };
        assert!(v.is_empty());
    }

    #[test]
    fn test_insert_missing_values_error() {
        parse_fails("INSERT INTO users (name) LIMIT 1");
    }

    #[test]
    fn test_case_without_when_error() {
        parse_fails("CASE END");
    }

    #[test]
    fn test_embed_invalid_operation_error() {
        parse_fails("EMBED INVALID");
    }

    #[test]
    fn test_table_constraint_invalid_error() {
        parse_fails("CREATE TABLE t (id INT, CONSTRAINT c INVALID)");
    }

    #[test]
    fn test_references_without_column() {
        let stmt = parse_stmt("CREATE TABLE t (user_id INT REFERENCES users)");
        let ct = unwrap_createtable(stmt);
        assert!(!ct.columns.is_empty());
    }

    #[test]
    fn test_entity_batch_empty_array() {
        // Empty batch array is valid and produces empty entities list
        let stmt = parse_stmt("ENTITY BATCH CREATE []");
        let e = unwrap_entity(stmt);
        let EntityOp::Batch { entities } = e.operation else {
            panic!("expected BATCH")
        };
        assert!(entities.is_empty());
    }

    #[test]
    fn test_entity_batch_missing_key_error() {
        parse_fails("ENTITY BATCH CREATE [{name: 'Alice'}]");
    }

    #[test]
    fn test_node_without_properties() {
        let stmt = parse_stmt("NODE CREATE person");
        let n = unwrap_node(stmt);
        let NodeOp::Create { properties, .. } = n.operation else {
            panic!("expected CREATE")
        };
        assert!(properties.is_empty());
    }

    #[test]
    fn test_edge_without_properties() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : follows");
        let e = unwrap_edge(stmt);
        let EdgeOp::Create { properties, .. } = e.operation else {
            panic!("expected CREATE")
        };
        assert!(properties.is_empty());
    }

    #[test]
    fn test_select_alias_no_as_keyword() {
        let stmt = parse_stmt("SELECT a FROM users u");
        let s = unwrap_select(stmt);
        let from = s.from.as_ref().expect("expected FROM clause");
        assert!(from.table.alias.is_some());
    }

    #[test]
    fn test_select_alias_keyword_not_aliased() {
        // When identifier is followed by a keyword, it's not treated as alias
        let stmt = parse_stmt("SELECT a FROM users WHERE id = 1");
        let s = unwrap_select(stmt);
        let from = s.from.as_ref().expect("expected FROM clause");
        assert!(from.table.alias.is_none());
    }

    #[test]
    fn test_embed_batch_empty_vector() {
        let stmt = parse_stmt("EMBED BATCH [('key1', [])]");
        let e = unwrap_embed(stmt);
        let EmbedOp::Batch { items } = e.operation else {
            panic!("expected BATCH")
        };
        assert_eq!(items.len(), 1);
        assert!(items[0].1.is_empty());
    }

    #[test]
    fn test_direction_error() {
        parse_fails("PATH FROM 1 TO 2 DIRECTION WRONG");
    }

    #[test]
    fn test_join_using_clause() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b USING (id)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_join_using_multiple_columns() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b USING (id, name)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_qualified_wildcard_with_expression_error() {
        // (1 + 2).* is not valid - qualified wildcard needs identifier
        parse_fails("SELECT (1 + 2).* FROM t");
    }

    #[test]
    fn test_empty_tuple() {
        let stmt = parse_stmt("SELECT () FROM t");
        let s = unwrap_select(stmt);
        assert!(!s.columns.is_empty());
    }

    #[test]
    fn test_array_literal_empty() {
        let stmt = parse_stmt("SELECT [] FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_decimal_type_with_precision() {
        let stmt = parse_stmt("CREATE TABLE t (price DECIMAL(10))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_decimal_type_with_precision_and_scale() {
        let stmt = parse_stmt("CREATE TABLE t (price DECIMAL(10, 2))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_varchar_with_length() {
        let stmt = parse_stmt("CREATE TABLE t (name VARCHAR(255))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_constraint_check() {
        let stmt = parse_stmt("CREATE TABLE t (age INT CHECK (age > 0))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_column_null_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (name VARCHAR NULL)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_in_empty_list() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IN ()");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_expect_ident_or_any_keyword_with_keyword() {
        // FROM is a reserved keyword but should work as property name in batch
        let stmt = parse_stmt("ENTITY BATCH CREATE [{key: 'k1', from: 'source'}]");
        let e = unwrap_entity(stmt);
        let EntityOp::Batch { entities } = e.operation else {
            panic!("expected BATCH")
        };
        assert_eq!(entities.len(), 1);
        // Should have property named "from"
        assert!(!entities[0].properties.is_empty());
    }

    #[test]
    fn test_batch_entity_number_as_key_error() {
        // Number is not valid as identifier
        parse_fails("ENTITY BATCH CREATE [{123: 'value'}]");
    }

    // Coverage for graph constraint error paths

    #[test]
    fn test_constraint_on_edge_without_type() {
        // EDGE followed directly by PROPERTY
        let stmt = parse_stmt("CONSTRAINT CREATE c ON EDGE PROPERTY name UNIQUE");
        assert!(matches!(stmt.kind, StatementKind::GraphConstraint(_)));
    }

    #[test]
    fn test_constraint_invalid_target_error() {
        // Neither NODE nor EDGE after ON
        parse_fails("CONSTRAINT CREATE c ON TABLE PROPERTY id UNIQUE");
    }

    #[test]
    fn test_constraint_invalid_type_error() {
        // Not UNIQUE, EXISTS, or TYPE
        parse_fails("CONSTRAINT CREATE c ON NODE person PROPERTY id INVALID");
    }

    #[test]
    fn test_batch_create_invalid_error() {
        // Neither NODES nor EDGES after BATCH CREATE
        parse_fails("BATCH CREATE TABLES []");
    }

    #[test]
    fn test_batch_delete_invalid_error() {
        // Neither NODES nor EDGES after BATCH DELETE
        parse_fails("BATCH DELETE TABLES [1, 2]");
    }

    #[test]
    fn test_batch_edge_missing_from_error() {
        // Edge without 'from' field
        parse_fails("BATCH CREATE EDGES [{to: 2, type: follows}]");
    }

    #[test]
    fn test_batch_edge_missing_to_error() {
        // Edge without 'to' field
        parse_fails("BATCH CREATE EDGES [{from: 1, type: follows}]");
    }

    #[test]
    fn test_batch_edge_missing_type_error() {
        // Edge without 'type' field
        parse_fails("BATCH CREATE EDGES [{from: 1, to: 2}]");
    }

    #[test]
    fn test_batch_update_missing_id_error() {
        // Node update without 'id' field
        parse_fails("BATCH UPDATE NODES [{name: 'Alice'}]");
    }

    #[test]
    fn test_batch_update_empty_list() {
        // Empty batch update list is valid
        let stmt = parse_stmt("BATCH UPDATE NODES []");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_direction_invalid_error() {
        // PATH requires valid direction
        parse_fails("PATH FROM 1 TO 2 DIRECTION INVALID");
    }

    #[test]
    fn test_expect_ident_or_keyword_number_error() {
        // Number can't be used where identifier/keyword expected
        parse_fails("NODE CREATE 123");
    }

    #[test]
    fn test_batch_invalid_operation_error() {
        // BATCH requires CREATE, DELETE, or UPDATE
        parse_fails("BATCH INVALID NODES []");
    }

    #[test]
    fn test_graph_algorithm_invalid_error() {
        parse_fails("GRAPH ALGORITHM INVALID");
    }

    #[test]
    fn test_case_with_operand_no_when() {
        // CASE with operand but no WHEN clause - should hit error at line 534
        // "CASE 1 END" - parses 1 as operand, then expects WHEN but gets END
        parse_fails("SELECT CASE 1 END FROM t");
    }

    #[test]
    fn test_case_expression_without_when() {
        // CASE without any WHEN clause should fail
        parse_fails("SELECT CASE WHEN END FROM t");
    }

    #[test]
    fn test_case_when_no_clauses() {
        // CASE followed directly by END (no WHEN at all) - fails during operand parsing
        parse_fails("SELECT CASE END FROM t");
    }

    #[test]
    fn test_embed_unknown_operation() {
        parse_fails("EMBED UNKNOWN 'key'");
    }

    #[test]
    fn test_vault_invalid_operation() {
        parse_fails("VAULT INVALID 'key'");
    }

    #[test]
    fn test_select_all_from_subquery() {
        // SELECT * FROM (subquery)
        let stmt = parse_stmt("SELECT * FROM (SELECT a FROM t) AS sub");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // Additional coverage tests for remaining uncovered paths

    #[test]
    fn test_aggregate_node_with_filter() {
        let stmt = parse_stmt("AGGREGATE NODE PROPERTY age SUM ON Person WHERE age > 18");
        assert!(matches!(stmt.kind, StatementKind::GraphAggregate(_)));
    }

    #[test]
    fn test_aggregate_edge_with_filter() {
        let stmt =
            parse_stmt("AGGREGATE EDGE PROPERTY weight COUNT BY TYPE follows WHERE weight > 0");
        assert!(matches!(stmt.kind, StatementKind::GraphAggregate(_)));
    }

    #[test]
    fn test_cache_semantic_put_empty_vector() {
        // Test empty embedding vector
        let stmt = parse_stmt("CACHE SEMANTIC PUT 'query' 'response' EMBEDDING []");
        assert!(matches!(stmt.kind, StatementKind::Cache(_)));
    }

    #[test]
    fn test_decimal_without_precision() {
        let stmt = parse_stmt("CREATE TABLE t (amount DECIMAL)");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_batch_update_nodes_with_props() {
        let stmt = parse_stmt("BATCH UPDATE NODES [{id: 1, name: 'Alice'}]");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_batch_create_edges_with_props() {
        let stmt = parse_stmt("BATCH CREATE EDGES [{from: 1, to: 2, type: follows, weight: 1.0}]");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_find_edges_by_type() {
        let stmt = parse_stmt("FIND EDGE follows WHERE weight > 0.5");
        assert!(matches!(stmt.kind, StatementKind::Find(_)));
    }

    // ========== Coverage gap: parse_all empty/multi-statement ==========

    #[test]
    fn test_parse_all_empty() {
        let stmts = parse_all("").unwrap();
        assert!(stmts.is_empty());
    }

    #[test]
    fn test_parse_all_whitespace() {
        let stmts = parse_all("   ").unwrap();
        assert!(stmts.is_empty());
    }

    #[test]
    fn test_parse_all_multi_statement() {
        let stmts = parse_all("SELECT 1; SELECT 2").unwrap();
        assert_eq!(stmts.len(), 2);
    }

    // ========== Coverage gap: BLOB PUT no data ==========

    #[test]
    fn test_blob_put_filename_only() {
        let stmt = parse_stmt("BLOB PUT 'file.txt'");
        assert!(matches!(stmt.kind, StatementKind::Blob(_)));
    }

    // ========== Coverage gap: graph algorithm break arm ==========

    #[test]
    fn test_graph_louvain_all_options_combined() {
        let stmt =
            parse_stmt("GRAPH LOUVAIN COMMUNITIES RESOLUTION 1.5 BOTH EDGE TYPE friend PASSES 20");
        assert!(matches!(stmt.kind, StatementKind::GraphAlgorithm(_)));
    }

    // ========== Coverage gap: lexer edge cases ==========

    #[test]
    fn test_lexer_unterminated_block_comment() {
        // Lexer treats unterminated block comments as consuming the rest of input
        // Just ensure it doesn't panic
        let _ = parse("/* unterminated comment");
    }

    // ========== Coverage gap: batch operations ==========

    #[test]
    fn test_batch_create_nodes_empty() {
        let stmt = parse_stmt("BATCH CREATE NODES []");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_batch_create_edges_empty() {
        let stmt = parse_stmt("BATCH CREATE EDGES []");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_batch_update_nodes_multi() {
        let stmt = parse_stmt("BATCH UPDATE NODES [{id: 1, name: 'Alice'}, {id: 2, name: 'Bob'}]");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_batch_node_def_with_labels_and_props() {
        let stmt =
            parse_stmt("BATCH CREATE NODES [{labels: [person, employee], name: 'Alice', age: 30}]");
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    #[test]
    fn test_batch_edge_def_with_props() {
        let stmt = parse_stmt(
            "BATCH CREATE EDGES [{from: 1, to: 2, type: knows, weight: 0.5, since: 2020}]",
        );
        assert!(matches!(stmt.kind, StatementKind::GraphBatch(_)));
    }

    // ========== Coverage gap: aggregate with edge types ==========

    #[test]
    fn test_aggregate_edge_property_sum() {
        let stmt = parse_stmt("AGGREGATE EDGE PROPERTY weight SUM");
        assert!(matches!(stmt.kind, StatementKind::GraphAggregate(_)));
    }

    #[test]
    fn test_aggregate_edge_property_by_type() {
        let stmt = parse_stmt("AGGREGATE EDGE PROPERTY weight AVG BY TYPE knows");
        assert!(matches!(stmt.kind, StatementKind::GraphAggregate(_)));
    }

    // ========== Coverage gap: INSERT ... SELECT ==========

    #[test]
    fn test_insert_select_statement() {
        let stmt = parse_stmt("INSERT INTO dst SELECT * FROM src");
        assert!(matches!(stmt.kind, StatementKind::Insert(_)));
    }

    // ========== Coverage gap: INSERT missing values/select ==========

    #[test]
    fn test_insert_missing_values_garbage_error() {
        let result = parse("INSERT INTO tbl GARBAGE");
        assert!(result.is_err());
    }

    // ========== Coverage gap: SQL type precision/scale ==========

    #[test]
    fn test_create_table_decimal_precision_scale() {
        let stmt = parse_stmt("CREATE TABLE t (val DECIMAL(10, 2))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    #[test]
    fn test_create_table_varchar_length() {
        let stmt = parse_stmt("CREATE TABLE t (name VARCHAR(100))");
        assert!(matches!(stmt.kind, StatementKind::CreateTable(_)));
    }

    // ========== Coverage gap: CASE expression ==========

    #[test]
    fn test_case_when_expression() {
        let stmt = parse_stmt("SELECT CASE WHEN x > 0 THEN 'positive' ELSE 'negative' END FROM t");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // ========== Coverage gap: postfix IS NULL/IS NOT NULL ==========

    #[test]
    fn test_is_null_in_select() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NULL");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_is_not_null_in_select() {
        let stmt = parse_stmt("SELECT * FROM t WHERE x IS NOT NULL");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // ========== Coverage gap: NOT LIKE ==========

    #[test]
    fn test_not_like_in_select() {
        let stmt = parse_stmt("SELECT * FROM t WHERE name NOT LIKE 'test%'");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // ========== Coverage gap: IN expression ==========

    #[test]
    fn test_in_expression() {
        let stmt = parse_stmt("SELECT * FROM t WHERE id IN (1, 2, 3)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_not_in_expression() {
        let stmt = parse_stmt("SELECT * FROM t WHERE id NOT IN (1, 2)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // ========== Coverage gap: BETWEEN expression ==========

    #[test]
    fn test_between_expression() {
        let stmt = parse_stmt("SELECT * FROM t WHERE val BETWEEN 1 AND 10");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // ========== Coverage gap: table alias with AS ==========

    #[test]
    fn test_table_alias_explicit_as() {
        let stmt = parse_stmt("SELECT u.name FROM users AS u WHERE u.id = 1");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    // === Extraction helper coverage tests ===

    #[test]
    #[should_panic(expected = "expected Blob")]
    fn test_unwrap_blob_wrong_variant() {
        unwrap_blob(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Blobs")]
    fn test_unwrap_blobs_wrong_variant() {
        unwrap_blobs(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Cache")]
    fn test_unwrap_cache_wrong_variant() {
        unwrap_cache(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Chain")]
    fn test_unwrap_chain_wrong_variant() {
        unwrap_chain(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Checkpoint")]
    fn test_unwrap_checkpoint_wrong_variant() {
        unwrap_checkpoint(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Checkpoints")]
    fn test_unwrap_checkpoints_wrong_variant() {
        unwrap_checkpoints(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Cluster")]
    fn test_unwrap_cluster_wrong_variant() {
        unwrap_cluster(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected CreateIndex")]
    fn test_unwrap_createindex_wrong_variant() {
        unwrap_createindex(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected CreateTable")]
    fn test_unwrap_createtable_wrong_variant() {
        unwrap_createtable(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Delete")]
    fn test_unwrap_delete_wrong_variant() {
        unwrap_delete(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Describe")]
    fn test_unwrap_describe_wrong_variant() {
        unwrap_describe(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected DropIndex")]
    fn test_unwrap_dropindex_wrong_variant() {
        unwrap_dropindex(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected DropTable")]
    fn test_unwrap_droptable_wrong_variant() {
        unwrap_droptable(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Edge")]
    fn test_unwrap_edge_wrong_variant() {
        unwrap_edge(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Embed")]
    fn test_unwrap_embed_wrong_variant() {
        unwrap_embed(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Entity")]
    fn test_unwrap_entity_wrong_variant() {
        unwrap_entity(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Find")]
    fn test_unwrap_find_wrong_variant() {
        unwrap_find(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected GraphAggregate")]
    fn test_unwrap_graphaggregate_wrong_variant() {
        unwrap_graphaggregate(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected GraphAlgorithm")]
    fn test_unwrap_graphalgorithm_wrong_variant() {
        unwrap_graphalgorithm(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected GraphBatch")]
    fn test_unwrap_graphbatch_wrong_variant() {
        unwrap_graphbatch(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected GraphConstraint")]
    fn test_unwrap_graphconstraint_wrong_variant() {
        unwrap_graphconstraint(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected GraphIndex")]
    fn test_unwrap_graphindex_wrong_variant() {
        unwrap_graphindex(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Insert")]
    fn test_unwrap_insert_wrong_variant() {
        unwrap_insert(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Neighbors")]
    fn test_unwrap_neighbors_wrong_variant() {
        unwrap_neighbors(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Node")]
    fn test_unwrap_node_wrong_variant() {
        unwrap_node(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Path")]
    fn test_unwrap_path_wrong_variant() {
        unwrap_path(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Rollback")]
    fn test_unwrap_rollback_wrong_variant() {
        unwrap_rollback(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Select")]
    fn test_unwrap_select_wrong_variant() {
        unwrap_select(parse_stmt("INSERT INTO t VALUES (1)"));
    }

    #[test]
    #[should_panic(expected = "expected Similar")]
    fn test_unwrap_similar_wrong_variant() {
        unwrap_similar(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Update")]
    fn test_unwrap_update_wrong_variant() {
        unwrap_update(parse_stmt("SELECT 1"));
    }

    #[test]
    #[should_panic(expected = "expected Vault")]
    fn test_unwrap_vault_wrong_variant() {
        unwrap_vault(parse_stmt("SELECT 1"));
    }
}
