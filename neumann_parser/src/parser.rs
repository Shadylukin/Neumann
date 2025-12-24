//! Statement parser for the Neumann query language.
//!
//! Parses complete statements including:
//! - SQL statements (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP)
//! - Graph commands (NODE, EDGE, NEIGHBORS, PATH)
//! - Vector commands (EMBED, SIMILAR)
//! - Unified queries (FIND)

use crate::ast::*;
use crate::error::{ParseError, ParseErrorKind, ParseResult};
use crate::lexer::Lexer;
use crate::span::Span;
use crate::token::{Token, TokenKind};

/// Binding power for prefix operators.
const PREFIX_BP: u8 = 19;

/// Returns binding power for infix operators.
fn infix_binding_power(op: BinaryOp) -> (u8, u8) {
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
    pub fn source(&self) -> &'a str {
        self.source
    }

    /// Returns the current token.
    pub fn current(&self) -> &Token {
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

            TokenKind::Eof => Err(ParseError::unexpected_eof(token.span, "expression")),

            _ => Err(ParseError::unexpected(
                token.kind.clone(),
                token.span,
                "expression",
            )),
        }
    }

    /// Parses postfix operators.
    fn parse_postfix_expr(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        loop {
            // Check for NOT followed by IN/BETWEEN/LIKE
            if self.check(&TokenKind::Not) {
                let next_kind = self.peek().kind.clone();
                if next_kind == TokenKind::In {
                    self.advance();
                    expr = self.parse_in_expr(expr)?;
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
                TokenKind::In => self.parse_in_expr(expr)?,
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
        let ident = Ident::new(name.clone(), token.span);

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

    fn parse_in_expr(&mut self, expr: Expr) -> ParseResult<Expr> {
        let start_span = expr.span;
        self.expect(&TokenKind::In)?;
        self.expect(&TokenKind::LParen)?;
        let mut values = Vec::new();
        if !self.check(&TokenKind::RParen) {
            loop {
                values.push(self.parse_expr()?);
                if !self.eat(&TokenKind::Comma) {
                    break;
                }
            }
        }
        let end = self.expect(&TokenKind::RParen)?.span;
        Ok(Expr::new(
            ExprKind::In(Box::new(expr), InList::Values(values)),
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

    fn current_binary_op(&self) -> Option<BinaryOp> {
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

        Ok(StatementKind::Select(SelectStmt {
            distinct,
            columns,
            from,
            where_clause,
            group_by,
            having,
            order_by,
            limit,
            offset,
        }))
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

        // TODO: Support subqueries
        let name = self.expect_ident()?;
        let kind = TableRefKind::Table(name);

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
        let name = self.expect_ident()?;
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

            let name = self.expect_ident()?;

            Ok(StatementKind::DropIndex(DropIndexStmt { if_exists, name }))
        } else {
            Err(ParseError::unexpected(
                self.current.kind.clone(),
                self.current.span,
                "TABLE or INDEX",
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
            let label = if !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
                Some(self.expect_ident()?)
            } else {
                None
            };
            NodeOp::List { label }
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
            let edge_type = if !self.current.is_eof() && !self.check(&TokenKind::Semicolon) {
                Some(self.expect_ident()?)
            } else {
                None
            };
            EdgeOp::List { edge_type }
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
                let key = self.expect_ident()?;
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

        Ok(StatementKind::Neighbors(NeighborsStmt {
            node_id,
            direction,
            edge_type,
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
        } else {
            return Err(ParseError::invalid(
                "expected STORE, GET, or DELETE after EMBED",
                self.current.span,
            ));
        };

        Ok(StatementKind::Embed(EmbedStmt { operation }))
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

        Ok(StatementKind::Similar(SimilarStmt {
            query,
            limit,
            metric,
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
}

/// Parses a single statement from source text.
pub fn parse(source: &str) -> ParseResult<Statement> {
    let mut parser = Parser::new(source);
    parser.parse_statement()
}

/// Parses multiple statements from source text.
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
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.columns.len(), 3);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_with_alias() {
        let stmt = parse_stmt("SELECT name AS user_name FROM users");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.columns[0].alias.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_distinct() {
        let stmt = parse_stmt("SELECT DISTINCT name FROM users");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.distinct);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_where() {
        let stmt = parse_stmt("SELECT * FROM users WHERE id = 1");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.where_clause.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_order_by() {
        let stmt = parse_stmt("SELECT * FROM users ORDER BY name ASC");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.order_by.len(), 1);
            assert_eq!(select.order_by[0].direction, SortDirection::Asc);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_limit_offset() {
        let stmt = parse_stmt("SELECT * FROM users LIMIT 10 OFFSET 5");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.limit.is_some());
            assert!(select.offset.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_join() {
        let stmt = parse_stmt("SELECT * FROM users u JOIN orders o ON u.id = o.user_id");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.from.as_ref().unwrap().joins.len(), 1);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_left_join() {
        let stmt = parse_stmt("SELECT * FROM users LEFT JOIN orders ON users.id = orders.user_id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Left);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_group_by_having() {
        let stmt = parse_stmt("SELECT name, COUNT(*) FROM users GROUP BY name HAVING COUNT(*) > 1");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(!select.group_by.is_empty());
            assert!(select.having.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_insert() {
        let stmt =
            parse_stmt("INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com')");
        if let StatementKind::Insert(insert) = stmt.kind {
            assert_eq!(insert.table.name, "users");
            assert!(insert.columns.is_some());
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_update() {
        let stmt = parse_stmt("UPDATE users SET name = 'Bob' WHERE id = 1");
        if let StatementKind::Update(update) = stmt.kind {
            assert_eq!(update.table.name, "users");
            assert_eq!(update.assignments.len(), 1);
            assert!(update.where_clause.is_some());
        } else {
            panic!("expected UPDATE");
        }
    }

    #[test]
    fn test_delete() {
        let stmt = parse_stmt("DELETE FROM users WHERE id = 1");
        if let StatementKind::Delete(delete) = stmt.kind {
            assert_eq!(delete.table.name, "users");
            assert!(delete.where_clause.is_some());
        } else {
            panic!("expected DELETE");
        }
    }

    #[test]
    fn test_create_table() {
        let stmt =
            parse_stmt("CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100) NOT NULL)");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert_eq!(create.table.name, "users");
            assert_eq!(create.columns.len(), 2);
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_create_table_if_not_exists() {
        let stmt = parse_stmt("CREATE TABLE IF NOT EXISTS users (id INT)");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create.if_not_exists);
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_create_index() {
        let stmt = parse_stmt("CREATE INDEX idx_name ON users (name)");
        if let StatementKind::CreateIndex(create) = stmt.kind {
            assert_eq!(create.name.name, "idx_name");
            assert!(!create.unique);
        } else {
            panic!("expected CREATE INDEX");
        }
    }

    #[test]
    fn test_create_unique_index() {
        let stmt = parse_stmt("CREATE UNIQUE INDEX idx_email ON users (email)");
        if let StatementKind::CreateIndex(create) = stmt.kind {
            assert!(create.unique);
        } else {
            panic!("expected CREATE INDEX");
        }
    }

    #[test]
    fn test_drop_table() {
        let stmt = parse_stmt("DROP TABLE users");
        if let StatementKind::DropTable(drop) = stmt.kind {
            assert_eq!(drop.table.name, "users");
            assert!(!drop.if_exists);
        } else {
            panic!("expected DROP TABLE");
        }
    }

    #[test]
    fn test_drop_table_if_exists() {
        let stmt = parse_stmt("DROP TABLE IF EXISTS users CASCADE");
        if let StatementKind::DropTable(drop) = stmt.kind {
            assert!(drop.if_exists);
            assert!(drop.cascade);
        } else {
            panic!("expected DROP TABLE");
        }
    }

    #[test]
    fn test_drop_index() {
        let stmt = parse_stmt("DROP INDEX IF EXISTS idx_name");
        if let StatementKind::DropIndex(drop) = stmt.kind {
            assert!(drop.if_exists);
        } else {
            panic!("expected DROP INDEX");
        }
    }

    #[test]
    fn test_node_create() {
        let stmt = parse_stmt("NODE CREATE user {name: 'Alice', age: 30}");
        if let StatementKind::Node(node) = stmt.kind {
            if let NodeOp::Create { label, properties } = node.operation {
                assert_eq!(label.name, "user");
                assert_eq!(properties.len(), 2);
            } else {
                panic!("expected NODE CREATE");
            }
        } else {
            panic!("expected NODE");
        }
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
                operation: NodeOp::List { label: Some(_) }
            })
        ));
    }

    #[test]
    fn test_edge_create() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : knows {since: 2020}");
        if let StatementKind::Edge(edge) = stmt.kind {
            if let EdgeOp::Create { edge_type, .. } = edge.operation {
                assert_eq!(edge_type.name, "knows");
            } else {
                panic!("expected EDGE CREATE");
            }
        } else {
            panic!("expected EDGE");
        }
    }

    #[test]
    fn test_neighbors() {
        let stmt = parse_stmt("NEIGHBORS 1 OUTGOING");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert_eq!(neighbors.direction, Direction::Outgoing);
        } else {
            panic!("expected NEIGHBORS");
        }
    }

    #[test]
    fn test_path() {
        let stmt = parse_stmt("PATH SHORTEST 1 -> 2 LIMIT 5");
        if let StatementKind::Path(path) = stmt.kind {
            assert_eq!(path.algorithm, PathAlgorithm::Shortest);
            assert!(path.max_depth.is_some());
        } else {
            panic!("expected PATH");
        }
    }

    #[test]
    fn test_embed_store() {
        let stmt = parse_stmt("EMBED STORE 'doc1' [0.1, 0.2, 0.3]");
        if let StatementKind::Embed(embed) = stmt.kind {
            if let EmbedOp::Store { vector, .. } = embed.operation {
                assert_eq!(vector.len(), 3);
            } else {
                panic!("expected EMBED STORE");
            }
        } else {
            panic!("expected EMBED");
        }
    }

    #[test]
    fn test_embed_get() {
        let stmt = parse_stmt("EMBED GET 'doc1'");
        assert!(matches!(
            stmt.kind,
            StatementKind::Embed(EmbedStmt {
                operation: EmbedOp::Get { .. }
            })
        ));
    }

    #[test]
    fn test_similar() {
        let stmt = parse_stmt("SIMILAR 'doc1' LIMIT 10 COSINE");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert!(matches!(similar.query, SimilarQuery::Key(_)));
            assert!(similar.limit.is_some());
            assert_eq!(similar.metric, Some(DistanceMetric::Cosine));
        } else {
            panic!("expected SIMILAR");
        }
    }

    #[test]
    fn test_similar_vector() {
        let stmt = parse_stmt("SIMILAR [0.1, 0.2] LIMIT 5");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert!(matches!(similar.query, SimilarQuery::Vector(_)));
        } else {
            panic!("expected SIMILAR");
        }
    }

    #[test]
    fn test_find_nodes() {
        let stmt = parse_stmt("FIND NODE user WHERE age > 18 LIMIT 10");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(
                find.pattern,
                FindPattern::Nodes { label: Some(_) }
            ));
            assert!(find.where_clause.is_some());
            assert!(find.limit.is_some());
        } else {
            panic!("expected FIND");
        }
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
        if let StatementKind::CreateTable(create) = stmt.kind {
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
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_table_constraints() {
        let stmt =
            parse_stmt("CREATE TABLE t (id INT, name TEXT, PRIMARY KEY (id), UNIQUE (name))");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert_eq!(create.constraints.len(), 2);
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_join_using() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b USING (id)");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert!(matches!(
                from.joins[0].condition,
                Some(JoinCondition::Using(_))
            ));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_order_by_nulls() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x NULLS FIRST");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.order_by[0].nulls, Some(NullsOrder::First));
        } else {
            panic!("expected SELECT");
        }
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
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(matches!(
                create.columns[0].data_type,
                DataType::Char(Some(10))
            ));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_numeric_precision_scale() {
        let stmt = parse_stmt("CREATE TABLE t (x NUMERIC(5, 2))");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(matches!(
                create.columns[0].data_type,
                DataType::Numeric(Some(5), Some(2))
            ));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_right_join() {
        let stmt = parse_stmt("SELECT * FROM a RIGHT JOIN b ON a.id = b.id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Right);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_full_join() {
        let stmt = parse_stmt("SELECT * FROM a FULL OUTER JOIN b ON a.id = b.id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Full);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_cross_join() {
        let stmt = parse_stmt("SELECT * FROM a CROSS JOIN b");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Cross);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_natural_join() {
        let stmt = parse_stmt("SELECT * FROM a NATURAL JOIN b");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Natural);
        } else {
            panic!("expected SELECT");
        }
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
        if let StatementKind::Edge(edge) = stmt.kind {
            if let EdgeOp::List { edge_type } = edge.operation {
                assert!(edge_type.is_some());
            } else {
                panic!("expected EDGE LIST");
            }
        } else {
            panic!("expected EDGE");
        }
    }

    #[test]
    fn test_edge_list_no_type() {
        let stmt = parse_stmt("EDGE LIST");
        if let StatementKind::Edge(edge) = stmt.kind {
            if let EdgeOp::List { edge_type } = edge.operation {
                assert!(edge_type.is_none());
            } else {
                panic!("expected EDGE LIST");
            }
        } else {
            panic!("expected EDGE");
        }
    }

    #[test]
    fn test_node_list_no_label() {
        let stmt = parse_stmt("NODE LIST");
        if let StatementKind::Node(node) = stmt.kind {
            if let NodeOp::List { label } = node.operation {
                assert!(label.is_none());
            } else {
                panic!("expected NODE LIST");
            }
        } else {
            panic!("expected NODE");
        }
    }

    #[test]
    fn test_embed_delete() {
        let stmt = parse_stmt("EMBED DELETE 'doc1'");
        assert!(matches!(
            stmt.kind,
            StatementKind::Embed(EmbedStmt {
                operation: EmbedOp::Delete { .. }
            })
        ));
    }

    #[test]
    fn test_neighbors_incoming() {
        let stmt = parse_stmt("NEIGHBORS 1 INCOMING");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert_eq!(neighbors.direction, Direction::Incoming);
        } else {
            panic!("expected NEIGHBORS");
        }
    }

    #[test]
    fn test_neighbors_both() {
        let stmt = parse_stmt("NEIGHBORS 1 BOTH");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert_eq!(neighbors.direction, Direction::Both);
        } else {
            panic!("expected NEIGHBORS");
        }
    }

    #[test]
    fn test_neighbors_with_type() {
        let stmt = parse_stmt("NEIGHBORS 1 OUTGOING : FOLLOWS");
        if let StatementKind::Neighbors(neighbors) = stmt.kind {
            assert!(neighbors.edge_type.is_some());
        } else {
            panic!("expected NEIGHBORS");
        }
    }

    #[test]
    fn test_path_without_shortest() {
        let stmt = parse_stmt("PATH 1 -> 2");
        if let StatementKind::Path(path) = stmt.kind {
            assert_eq!(path.algorithm, PathAlgorithm::Shortest);
        } else {
            panic!("expected PATH");
        }
    }

    #[test]
    fn test_path_with_limit() {
        let stmt = parse_stmt("PATH 1 -> 2 LIMIT 5");
        if let StatementKind::Path(path) = stmt.kind {
            assert!(path.max_depth.is_some());
        } else {
            panic!("expected PATH");
        }
    }

    #[test]
    fn test_find_edge() {
        let stmt = parse_stmt("FIND EDGE FOLLOWS WHERE weight > 0.5");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(
                find.pattern,
                FindPattern::Edges { edge_type: Some(_) }
            ));
        } else {
            panic!("expected FIND");
        }
    }

    #[test]
    fn test_find_vertex() {
        let stmt = parse_stmt("FIND VERTEX person");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(
                find.pattern,
                FindPattern::Nodes { label: Some(_) }
            ));
        } else {
            panic!("expected FIND");
        }
    }

    #[test]
    fn test_find_with_return() {
        let stmt = parse_stmt("FIND NODE user RETURN name, age");
        if let StatementKind::Find(find) = stmt.kind {
            assert_eq!(find.return_items.len(), 2);
        } else {
            panic!("expected FIND");
        }
    }

    #[test]
    fn test_find_no_pattern() {
        let stmt = parse_stmt("FIND WHERE x > 1");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(find.pattern, FindPattern::Nodes { label: None }));
        } else {
            panic!("expected FIND");
        }
    }

    #[test]
    fn test_similar_euclidean() {
        let stmt = parse_stmt("SIMILAR 'doc' EUCLIDEAN");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert_eq!(similar.metric, Some(DistanceMetric::Euclidean));
        } else {
            panic!("expected SIMILAR");
        }
    }

    #[test]
    fn test_similar_dot_product() {
        let stmt = parse_stmt("SIMILAR 'doc' DOT_PRODUCT");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert_eq!(similar.metric, Some(DistanceMetric::DotProduct));
        } else {
            panic!("expected SIMILAR");
        }
    }

    #[test]
    fn test_column_not_null() {
        let stmt = parse_stmt("CREATE TABLE t (x INT NOT NULL)");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create.columns[0]
                .constraints
                .iter()
                .any(|c| *c == ColumnConstraint::NotNull));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_column_default() {
        let stmt = parse_stmt("CREATE TABLE t (x INT DEFAULT 0)");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create.columns[0]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::Default(_))));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_column_references() {
        let stmt = parse_stmt("CREATE TABLE t (x INT REFERENCES other(id))");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create.columns[0]
                .constraints
                .iter()
                .any(|c| matches!(c, ColumnConstraint::References { .. })));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_insert_multiple_rows() {
        let stmt = parse_stmt("INSERT INTO t (a, b) VALUES (1, 2), (3, 4)");
        if let StatementKind::Insert(insert) = stmt.kind {
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
    fn test_insert_with_select() {
        let stmt = parse_stmt("INSERT INTO t SELECT * FROM other");
        if let StatementKind::Insert(insert) = stmt.kind {
            assert!(matches!(insert.source, InsertSource::Query(_)));
        } else {
            panic!("expected INSERT");
        }
    }

    #[test]
    fn test_order_by_nulls_last() {
        let stmt = parse_stmt("SELECT * FROM t ORDER BY x DESC NULLS LAST");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.order_by[0].direction, SortDirection::Desc);
            assert_eq!(select.order_by[0].nulls, Some(NullsOrder::Last));
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_multiple_columns() {
        let stmt = parse_stmt("SELECT a, b, c FROM t");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.columns.len(), 3);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_all_keyword() {
        let stmt = parse_stmt("SELECT ALL * FROM t");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(!select.distinct); // ALL is the default (not distinct)
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_inner_join() {
        let stmt = parse_stmt("SELECT * FROM a INNER JOIN b ON a.id = b.id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Inner);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_right_outer_join() {
        let stmt = parse_stmt("SELECT * FROM a RIGHT OUTER JOIN b ON a.id = b.id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Right);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_left_outer_join() {
        let stmt = parse_stmt("SELECT * FROM a LEFT OUTER JOIN b ON a.id = b.id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins[0].kind, JoinKind::Left);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_table_alias_implicit() {
        let stmt = parse_stmt("SELECT * FROM users u");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.table.alias.as_ref().unwrap().name, "u");
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_select_item_alias() {
        let stmt = parse_stmt("SELECT x AS y FROM t");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.columns[0].alias.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_foreign_key_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, FOREIGN KEY (x) REFERENCES other(id))");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create
                .constraints
                .iter()
                .any(|c| matches!(c, TableConstraint::ForeignKey { .. })));
        } else {
            panic!("expected CREATE TABLE");
        }
    }

    #[test]
    fn test_check_constraint() {
        let stmt = parse_stmt("CREATE TABLE t (x INT, CHECK (x > 0))");
        if let StatementKind::CreateTable(create) = stmt.kind {
            assert!(create
                .constraints
                .iter()
                .any(|c| matches!(c, TableConstraint::Check(_))));
        } else {
            panic!("expected CREATE TABLE");
        }
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
        if let StatementKind::Node(node) = stmt.kind {
            if let NodeOp::Create { properties, .. } = node.operation {
                assert!(properties.is_empty());
            } else {
                panic!("expected NODE CREATE");
            }
        } else {
            panic!("expected NODE");
        }
    }

    #[test]
    fn test_edge_create_empty_properties() {
        let stmt = parse_stmt("EDGE CREATE 1 -> 2 : knows {}");
        if let StatementKind::Edge(edge) = stmt.kind {
            if let EdgeOp::Create { properties, .. } = edge.operation {
                assert!(properties.is_empty());
            } else {
                panic!("expected EDGE CREATE");
            }
        } else {
            panic!("expected EDGE");
        }
    }

    // More SQL tests
    #[test]
    fn test_update_multiple_columns() {
        let stmt = parse_stmt("UPDATE t SET a = 1, b = 2, c = 3 WHERE id = 1");
        if let StatementKind::Update(update) = stmt.kind {
            assert_eq!(update.assignments.len(), 3);
        } else {
            panic!("expected UPDATE");
        }
    }

    #[test]
    fn test_delete_without_where() {
        let stmt = parse_stmt("DELETE FROM users");
        if let StatementKind::Delete(delete) = stmt.kind {
            assert!(delete.where_clause.is_none());
        } else {
            panic!("expected DELETE");
        }
    }

    #[test]
    fn test_select_with_offset() {
        let stmt = parse_stmt("SELECT * FROM t LIMIT 10 OFFSET 5");
        if let StatementKind::Select(select) = stmt.kind {
            assert!(select.limit.is_some());
            assert!(select.offset.is_some());
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_multiple_joins() {
        let stmt = parse_stmt("SELECT * FROM a JOIN b ON a.id = b.a_id JOIN c ON b.id = c.b_id");
        if let StatementKind::Select(select) = stmt.kind {
            let from = select.from.unwrap();
            assert_eq!(from.joins.len(), 2);
        } else {
            panic!("expected SELECT");
        }
    }

    #[test]
    fn test_complex_where_clause() {
        let stmt = parse_stmt("SELECT * FROM t WHERE (a > 1 AND b < 2) OR (c = 3 AND d != 4)");
        assert!(matches!(stmt.kind, StatementKind::Select(_)));
    }

    #[test]
    fn test_aggregate_with_filter() {
        let stmt = parse_stmt("SELECT COUNT(*), SUM(amount), AVG(price) FROM orders");
        if let StatementKind::Select(select) = stmt.kind {
            assert_eq!(select.columns.len(), 3);
        } else {
            panic!("expected SELECT");
        }
    }

    // Vector tests
    #[test]
    fn test_similar_no_options() {
        let stmt = parse_stmt("SIMILAR 'query'");
        if let StatementKind::Similar(similar) = stmt.kind {
            assert!(similar.limit.is_none());
            assert!(similar.metric.is_none());
        } else {
            panic!("expected SIMILAR");
        }
    }

    // FIND tests
    #[test]
    fn test_find_edge_no_type() {
        let stmt = parse_stmt("FIND EDGE WHERE weight > 0.5");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(
                find.pattern,
                FindPattern::Edges { edge_type: None }
            ));
        } else {
            panic!("expected FIND");
        }
    }

    #[test]
    fn test_find_node_no_label() {
        let stmt = parse_stmt("FIND NODE WHERE active = TRUE");
        if let StatementKind::Find(find) = stmt.kind {
            assert!(matches!(find.pattern, FindPattern::Nodes { label: None }));
        } else {
            panic!("expected FIND");
        }
    }
}
