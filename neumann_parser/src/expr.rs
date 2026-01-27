//! Pratt expression parser for the Neumann query language.
//!
//! Implements a Pratt parser (top-down operator precedence parser) for
//! parsing expressions with correct operator precedence and associativity.
//!
//! Precedence levels (higher binds tighter):
//! 1. OR
//! 2. AND
//! 3. Comparison (=, !=, <, <=, >, >=)
//! 4. Bitwise OR (|)
//! 5. Bitwise XOR (^)
//! 6. Bitwise AND (&)
//! 7. Shift (<<, >>)
//! 8. Additive (+, -, ||)
//! 9. Multiplicative (*, /, %)
//! 10. Unary (NOT, -, ~)
//! 11. Postfix (function calls, IS NULL, IN, BETWEEN, LIKE)

#![allow(clippy::wildcard_imports)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::if_not_else)]
#![allow(clippy::enum_glob_use)]

use crate::{
    ast::*,
    error::{ParseError, ParseResult},
    lexer::Lexer,
    span::Span,
    token::{Token, TokenKind},
};

/// Maximum expression nesting depth.
const MAX_DEPTH: usize = 64;

/// Expression parser using Pratt parsing.
pub struct ExprParser<'a> {
    lexer: Lexer<'a>,
    current: Token,
    peeked: Option<Token>,
    depth: usize,
}

impl<'a> ExprParser<'a> {
    /// Creates a new expression parser.
    pub fn new(source: &'a str) -> Self {
        let mut lexer = Lexer::new(source);
        let current = lexer.next_token();
        Self {
            lexer,
            current,
            peeked: None,
            depth: 0,
        }
    }

    /// Creates a parser from an existing lexer and current token.
    pub fn from_lexer(lexer: Lexer<'a>, current: Token) -> Self {
        Self {
            lexer,
            current,
            peeked: None,
            depth: 0,
        }
    }

    /// Returns the source text.
    pub fn source(&self) -> &'a str {
        self.lexer.source()
    }

    /// Returns the current token.
    pub fn current(&self) -> &Token {
        &self.current
    }

    /// Peeks at the next token.
    pub fn peek(&mut self) -> &Token {
        if self.peeked.is_none() {
            self.peeked = Some(self.lexer.next_token());
        }
        self.peeked.as_ref().unwrap()
    }

    /// Advances to the next token, returning the previous one.
    pub fn advance(&mut self) -> Token {
        std::mem::replace(
            &mut self.current,
            self.peeked
                .take()
                .unwrap_or_else(|| self.lexer.next_token()),
        )
    }

    /// Returns true if the current token matches the given kind.
    pub fn check(&self, kind: &TokenKind) -> bool {
        std::mem::discriminant(&self.current.kind) == std::mem::discriminant(kind)
    }

    /// Returns true if the current token is one of the given kinds.
    pub fn check_any(&self, kinds: &[TokenKind]) -> bool {
        kinds.iter().any(|k| self.check(k))
    }

    /// Consumes the current token if it matches.
    pub fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Expects the current token to match, or returns an error.
    pub fn expect(&mut self, kind: &TokenKind) -> ParseResult<Token> {
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

    /// Parses an expression.
    pub fn parse_expr(&mut self) -> ParseResult<Expr> {
        self.parse_expr_bp(0)
    }

    /// Parses an expression with the given minimum binding power.
    fn parse_expr_bp(&mut self, min_bp: u8) -> ParseResult<Expr> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ParseError::new(
                crate::error::ParseErrorKind::TooDeep,
                self.current.span,
            ));
        }

        let mut lhs = self.parse_prefix()?;

        loop {
            // Check for postfix operators
            lhs = self.parse_postfix(lhs)?;

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

        self.depth -= 1;
        Ok(lhs)
    }

    /// Parses a prefix expression (literals, identifiers, unary ops, parentheses).
    fn parse_prefix(&mut self) -> ParseResult<Expr> {
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

            // Identifiers and function calls
            TokenKind::Ident(_) => self.parse_ident_or_call(),

            // Aggregate functions
            TokenKind::Count
            | TokenKind::Sum
            | TokenKind::Avg
            | TokenKind::Min
            | TokenKind::Max => self.parse_aggregate_call(),

            // Wildcard
            TokenKind::Star => {
                self.advance();
                Ok(Expr::new(ExprKind::Wildcard, token.span))
            },

            // Parenthesized expression or tuple
            TokenKind::LParen => self.parse_paren_expr(),

            // Array literal
            TokenKind::LBracket => self.parse_array(),

            // Unary operators
            TokenKind::Minus => {
                self.advance();
                let operand = self.parse_expr_bp(prefix_binding_power())?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::Neg, Box::new(operand)),
                    span,
                ))
            },
            TokenKind::Not | TokenKind::Bang => {
                self.advance();
                let operand = self.parse_expr_bp(prefix_binding_power())?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::Not, Box::new(operand)),
                    span,
                ))
            },
            TokenKind::Tilde => {
                self.advance();
                let operand = self.parse_expr_bp(prefix_binding_power())?;
                let span = token.span.merge(operand.span);
                Ok(Expr::new(
                    ExprKind::Unary(UnaryOp::BitNot, Box::new(operand)),
                    span,
                ))
            },

            // CASE expression
            TokenKind::Case => self.parse_case(),

            // EXISTS subquery
            TokenKind::Exists => self.parse_exists(),

            TokenKind::Eof => Err(ParseError::unexpected_eof(token.span, "expression")),

            // Allow contextual keywords to be used as identifiers (e.g., column names like
            // "status")
            _ if token.kind.is_contextual_keyword() => self.parse_keyword_as_ident(),

            _ => Err(ParseError::unexpected(
                token.kind.clone(),
                token.span,
                "expression",
            )),
        }
    }

    /// Parses a keyword token as an identifier expression.
    fn parse_keyword_as_ident(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = token.kind.as_str().to_lowercase();
        let ident = Ident::new(name, token.span);
        Ok(Expr::new(ExprKind::Ident(ident), token.span))
    }

    /// Parses postfix operators (IS NULL, IN, BETWEEN, LIKE).
    fn parse_postfix(&mut self, mut expr: Expr) -> ParseResult<Expr> {
        loop {
            // Check for NOT followed by IN/BETWEEN/LIKE
            if self.check(&TokenKind::Not) {
                let next_kind = self.peek().kind.clone();
                if next_kind == TokenKind::In {
                    self.advance(); // consume NOT
                    expr = self.parse_in_expr(expr, true)?;
                    continue;
                } else if next_kind == TokenKind::Between {
                    self.advance(); // consume NOT
                    expr = self.parse_between_expr(expr, true)?;
                    continue;
                } else if next_kind == TokenKind::Like {
                    self.advance(); // consume NOT
                    expr = self.parse_like_expr(expr, true)?;
                    continue;
                }
            }

            let kind = self.current.kind.clone();
            expr = match kind {
                // IS [NOT] NULL
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

                // IN (values) or IN (subquery)
                TokenKind::In => self.parse_in_expr(expr, false)?,

                // BETWEEN low AND high
                TokenKind::Between => self.parse_between_expr(expr, false)?,

                // LIKE pattern
                TokenKind::Like => self.parse_like_expr(expr, false)?,

                // Qualified name (table.column or table.*)
                TokenKind::Dot => {
                    self.advance();
                    if self.eat(&TokenKind::Star) {
                        // table.*
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
                        // table.column
                        let token = self.expect(&TokenKind::Ident(String::new()))?;
                        let name = match token.kind {
                            TokenKind::Ident(s) => s,
                            _ => unreachable!(),
                        };
                        let span = expr.span.merge(token.span);
                        let ident = Ident::new(name, token.span);
                        Expr::new(ExprKind::Qualified(Box::new(expr), ident), span)
                    }
                },

                _ => return Ok(expr),
            };
        }
    }

    /// Parses an identifier or function call.
    fn parse_ident_or_call(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = match token.kind {
            TokenKind::Ident(s) => s,
            _ => unreachable!(),
        };
        let ident = Ident::new(name.clone(), token.span);

        // Check for function call
        if self.check(&TokenKind::LParen) {
            self.parse_function_call(ident, token.span)
        } else {
            Ok(Expr::new(ExprKind::Ident(ident), token.span))
        }
    }

    /// Parses a function call.
    fn parse_function_call(&mut self, name: Ident, start: Span) -> ParseResult<Expr> {
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
        let span = start.merge(end.span);

        Ok(Expr::new(
            ExprKind::Call(FunctionCall {
                name,
                args,
                distinct,
            }),
            span,
        ))
    }

    /// Parses an aggregate function call (COUNT, SUM, AVG, MIN, MAX).
    fn parse_aggregate_call(&mut self) -> ParseResult<Expr> {
        let token = self.advance();
        let name = match &token.kind {
            TokenKind::Count => "COUNT",
            TokenKind::Sum => "SUM",
            TokenKind::Avg => "AVG",
            TokenKind::Min => "MIN",
            TokenKind::Max => "MAX",
            _ => unreachable!(),
        };
        let ident = Ident::new(name, token.span);
        self.parse_function_call(ident, token.span)
    }

    /// Parses a parenthesized expression or tuple.
    fn parse_paren_expr(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::LParen)?.span;

        // Check for empty tuple
        if self.check(&TokenKind::RParen) {
            let end = self.advance().span;
            return Ok(Expr::new(ExprKind::Tuple(Vec::new()), start.merge(end)));
        }

        // TODO: Check for subquery (SELECT)

        let first = self.parse_expr()?;

        // Check for tuple
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
        // Return inner expression with updated span
        Ok(Expr::new(first.kind, start.merge(end)))
    }

    /// Parses an array literal.
    fn parse_array(&mut self) -> ParseResult<Expr> {
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

    /// Parses a CASE expression.
    fn parse_case(&mut self) -> ParseResult<Expr> {
        let start = self.expect(&TokenKind::Case)?.span;

        // Check for simple CASE (CASE expr WHEN ...)
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

    /// Parses an EXISTS subquery.
    fn parse_exists(&mut self) -> ParseResult<Expr> {
        let _start = self.expect(&TokenKind::Exists)?.span;
        // For now, just parse a parenthesized expression
        // Full subquery support requires the statement parser
        let token = self.expect(&TokenKind::LParen)?;
        Err(ParseError::invalid(
            "EXISTS subqueries not yet implemented",
            token.span,
        ))
    }

    /// Parses an IN expression.
    fn parse_in_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        self.expect(&TokenKind::In)?;
        self.expect(&TokenKind::LParen)?;

        // TODO: Check for subquery (SELECT)

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
        let span = expr.span.merge(end);

        Ok(Expr::new(
            ExprKind::In {
                expr: Box::new(expr),
                list: InList::Values(values),
                negated,
            },
            span,
        ))
    }

    /// Parses a BETWEEN expression.
    fn parse_between_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        self.expect(&TokenKind::Between)?;
        let low = self.parse_expr_bp(prefix_binding_power())?;
        self.expect(&TokenKind::And)?;
        let high = self.parse_expr_bp(prefix_binding_power())?;

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

    /// Parses a LIKE expression.
    fn parse_like_expr(&mut self, expr: Expr, negated: bool) -> ParseResult<Expr> {
        self.expect(&TokenKind::Like)?;
        let pattern = self.parse_expr_bp(prefix_binding_power())?;

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

    /// Returns the binary operator for the current token, if any.
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
}

/// Returns the binding power for infix operators (left and right).
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

/// Returns the binding power for prefix operators.
fn prefix_binding_power() -> u8 {
    19
}

/// Parses an expression from source text.
pub fn parse_expr(source: &str) -> ParseResult<Expr> {
    let mut parser = ExprParser::new(source);
    let expr = parser.parse_expr()?;

    // Ensure we consumed all input
    if !parser.current().is_eof() {
        return Err(ParseError::unexpected(
            parser.current().kind.clone(),
            parser.current().span,
            "end of expression",
        ));
    }

    Ok(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Expr {
        parse_expr(source).expect("parse failed")
    }

    fn parse_err(source: &str) -> ParseError {
        parse_expr(source).expect_err("expected parse error")
    }

    #[test]
    fn test_integer_literal() {
        let expr = parse("42");
        assert!(matches!(expr.kind, ExprKind::Literal(Literal::Integer(42))));
    }

    #[test]
    fn test_float_literal() {
        let expr = parse("3.14");
        assert!(matches!(
            expr.kind,
            ExprKind::Literal(Literal::Float(n)) if (n - 3.14).abs() < 0.001
        ));
    }

    #[test]
    fn test_string_literal() {
        let expr = parse("'hello'");
        assert!(matches!(
            expr.kind,
            ExprKind::Literal(Literal::String(ref s)) if s == "hello"
        ));
    }

    #[test]
    fn test_boolean_literals() {
        assert!(matches!(
            parse("TRUE").kind,
            ExprKind::Literal(Literal::Boolean(true))
        ));
        assert!(matches!(
            parse("FALSE").kind,
            ExprKind::Literal(Literal::Boolean(false))
        ));
    }

    #[test]
    fn test_null_literal() {
        assert!(matches!(
            parse("NULL").kind,
            ExprKind::Literal(Literal::Null)
        ));
    }

    #[test]
    fn test_identifier() {
        let expr = parse("foo");
        assert!(matches!(
            expr.kind,
            ExprKind::Ident(Ident { ref name, .. }) if name == "foo"
        ));
    }

    #[test]
    fn test_qualified_name() {
        let expr = parse("users.name");
        assert!(matches!(expr.kind, ExprKind::Qualified(_, ref ident) if ident.name == "name"));
    }

    #[test]
    fn test_wildcard() {
        assert!(matches!(parse("*").kind, ExprKind::Wildcard));
    }

    #[test]
    fn test_qualified_wildcard() {
        let expr = parse("users.*");
        assert!(matches!(
            expr.kind,
            ExprKind::QualifiedWildcard(ref ident) if ident.name == "users"
        ));
    }

    #[test]
    fn test_binary_arithmetic() {
        let expr = parse("1 + 2");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Add, _)));
    }

    #[test]
    fn test_operator_precedence() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let expr = parse("1 + 2 * 3");
        if let ExprKind::Binary(lhs, BinaryOp::Add, rhs) = expr.kind {
            assert!(matches!(lhs.kind, ExprKind::Literal(Literal::Integer(1))));
            assert!(matches!(rhs.kind, ExprKind::Binary(_, BinaryOp::Mul, _)));
        } else {
            panic!("expected binary add");
        }
    }

    #[test]
    fn test_left_associativity() {
        // 1 - 2 - 3 should parse as (1 - 2) - 3
        let expr = parse("1 - 2 - 3");
        if let ExprKind::Binary(lhs, BinaryOp::Sub, rhs) = expr.kind {
            assert!(matches!(lhs.kind, ExprKind::Binary(_, BinaryOp::Sub, _)));
            assert!(matches!(rhs.kind, ExprKind::Literal(Literal::Integer(3))));
        } else {
            panic!("expected binary sub");
        }
    }

    #[test]
    fn test_parentheses() {
        // (1 + 2) * 3
        let expr = parse("(1 + 2) * 3");
        if let ExprKind::Binary(lhs, BinaryOp::Mul, _) = expr.kind {
            assert!(matches!(lhs.kind, ExprKind::Binary(_, BinaryOp::Add, _)));
        } else {
            panic!("expected binary mul");
        }
    }

    #[test]
    fn test_unary_minus() {
        let expr = parse("-42");
        assert!(matches!(expr.kind, ExprKind::Unary(UnaryOp::Neg, _)));
    }

    #[test]
    fn test_unary_not() {
        let expr = parse("NOT TRUE");
        assert!(matches!(expr.kind, ExprKind::Unary(UnaryOp::Not, _)));
    }

    #[test]
    fn test_comparison_operators() {
        assert!(matches!(
            parse("a = b").kind,
            ExprKind::Binary(_, BinaryOp::Eq, _)
        ));
        assert!(matches!(
            parse("a != b").kind,
            ExprKind::Binary(_, BinaryOp::Ne, _)
        ));
        assert!(matches!(
            parse("a < b").kind,
            ExprKind::Binary(_, BinaryOp::Lt, _)
        ));
        assert!(matches!(
            parse("a <= b").kind,
            ExprKind::Binary(_, BinaryOp::Le, _)
        ));
        assert!(matches!(
            parse("a > b").kind,
            ExprKind::Binary(_, BinaryOp::Gt, _)
        ));
        assert!(matches!(
            parse("a >= b").kind,
            ExprKind::Binary(_, BinaryOp::Ge, _)
        ));
    }

    #[test]
    fn test_logical_operators() {
        assert!(matches!(
            parse("a AND b").kind,
            ExprKind::Binary(_, BinaryOp::And, _)
        ));
        assert!(matches!(
            parse("a OR b").kind,
            ExprKind::Binary(_, BinaryOp::Or, _)
        ));
    }

    #[test]
    fn test_function_call() {
        let expr = parse("foo(1, 2, 3)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "foo");
            assert_eq!(call.args.len(), 3);
            assert!(!call.distinct);
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_function_no_args() {
        let expr = parse("now()");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "now");
            assert!(call.args.is_empty());
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_aggregate_count() {
        let expr = parse("COUNT(*)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "COUNT");
            assert_eq!(call.args.len(), 1);
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_aggregate_distinct() {
        let expr = parse("COUNT(DISTINCT id)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "COUNT");
            assert!(call.distinct);
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_is_null() {
        let expr = parse("x IS NULL");
        assert!(matches!(expr.kind, ExprKind::IsNull { negated: false, .. }));
    }

    #[test]
    fn test_is_not_null() {
        let expr = parse("x IS NOT NULL");
        assert!(matches!(expr.kind, ExprKind::IsNull { negated: true, .. }));
    }

    #[test]
    fn test_in_list() {
        let expr = parse("x IN (1, 2, 3)");
        if let ExprKind::In {
            list: InList::Values(values),
            negated,
            ..
        } = expr.kind
        {
            assert_eq!(values.len(), 3);
            assert!(!negated);
        } else {
            panic!("expected IN expression");
        }
    }

    #[test]
    fn test_between() {
        let expr = parse("x BETWEEN 1 AND 10");
        assert!(matches!(
            expr.kind,
            ExprKind::Between { negated: false, .. }
        ));
    }

    #[test]
    fn test_not_between() {
        let expr = parse("x NOT BETWEEN 1 AND 10");
        assert!(matches!(expr.kind, ExprKind::Between { negated: true, .. }));
    }

    #[test]
    fn test_like() {
        let expr = parse("name LIKE '%foo%'");
        assert!(matches!(expr.kind, ExprKind::Like { negated: false, .. }));
    }

    #[test]
    fn test_not_like() {
        let expr = parse("name NOT LIKE '%foo%'");
        assert!(matches!(expr.kind, ExprKind::Like { negated: true, .. }));
    }

    #[test]
    fn test_case_simple() {
        let expr = parse("CASE x WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END");
        if let ExprKind::Case(case) = expr.kind {
            assert!(case.operand.is_some());
            assert_eq!(case.when_clauses.len(), 2);
            assert!(case.else_clause.is_some());
        } else {
            panic!("expected CASE expression");
        }
    }

    #[test]
    fn test_case_searched() {
        let expr = parse("CASE WHEN x > 0 THEN 'positive' ELSE 'non-positive' END");
        if let ExprKind::Case(case) = expr.kind {
            assert!(case.operand.is_none());
            assert_eq!(case.when_clauses.len(), 1);
        } else {
            panic!("expected CASE expression");
        }
    }

    #[test]
    fn test_array_literal() {
        let expr = parse("[1, 2, 3]");
        if let ExprKind::Array(items) = expr.kind {
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_empty_array() {
        let expr = parse("[]");
        if let ExprKind::Array(items) = expr.kind {
            assert!(items.is_empty());
        } else {
            panic!("expected array");
        }
    }

    #[test]
    fn test_tuple() {
        let expr = parse("(1, 2, 3)");
        if let ExprKind::Tuple(items) = expr.kind {
            assert_eq!(items.len(), 3);
        } else {
            panic!("expected tuple");
        }
    }

    #[test]
    fn test_complex_expression() {
        let expr = parse("(a + b) * 2 > c AND d IS NOT NULL");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::And, _)));
    }

    #[test]
    fn test_bitwise_operators() {
        assert!(matches!(
            parse("a & b").kind,
            ExprKind::Binary(_, BinaryOp::BitAnd, _)
        ));
        assert!(matches!(
            parse("a | b").kind,
            ExprKind::Binary(_, BinaryOp::BitOr, _)
        ));
        assert!(matches!(
            parse("a ^ b").kind,
            ExprKind::Binary(_, BinaryOp::BitXor, _)
        ));
        assert!(matches!(
            parse("a << b").kind,
            ExprKind::Binary(_, BinaryOp::Shl, _)
        ));
        assert!(matches!(
            parse("a >> b").kind,
            ExprKind::Binary(_, BinaryOp::Shr, _)
        ));
    }

    #[test]
    fn test_concat_operator() {
        assert!(matches!(
            parse("a || b").kind,
            ExprKind::Binary(_, BinaryOp::Concat, _)
        ));
    }

    #[test]
    fn test_error_unexpected_token() {
        let err = parse_err("1 + + 2");
        assert!(matches!(
            err.kind,
            crate::error::ParseErrorKind::UnexpectedToken { .. }
        ));
    }

    #[test]
    fn test_error_unexpected_eof() {
        let err = parse_err("1 +");
        assert!(matches!(
            err.kind,
            crate::error::ParseErrorKind::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn test_error_unclosed_paren() {
        let err = parse_err("(1 + 2");
        assert!(matches!(
            err.kind,
            crate::error::ParseErrorKind::UnexpectedEof { .. }
        ));
    }

    #[test]
    fn test_nested_function_calls() {
        let expr = parse("foo(bar(1), baz(2, 3))");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "foo");
            assert_eq!(call.args.len(), 2);
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_parser_methods() {
        let mut parser = ExprParser::new("1 + 2");

        assert!(parser.check(&TokenKind::Integer(0)));
        assert!(!parser.current().is_eof());
        assert_eq!(parser.source(), "1 + 2");

        // Test peek
        let peeked = parser.peek();
        assert!(matches!(peeked.kind, TokenKind::Plus));

        // Test advance
        let token = parser.advance();
        assert!(matches!(token.kind, TokenKind::Integer(1)));
    }

    #[test]
    fn test_check_any() {
        let parser = ExprParser::new("42");
        assert!(parser.check_any(&[TokenKind::Integer(0), TokenKind::Float(0.0)]));
        assert!(!parser.check_any(&[TokenKind::String(String::new())]));
    }

    #[test]
    fn test_case_no_when() {
        // "CASE END" tries to parse END as an expression (operand), which fails
        let err = parse_err("CASE END");
        assert!(matches!(
            err.kind,
            crate::error::ParseErrorKind::UnexpectedToken { .. }
        ));
    }

    #[test]
    fn test_deeply_nested() {
        // Build a deeply nested expression
        let mut expr = "1".to_string();
        for _ in 0..100 {
            expr = format!("({})", expr);
        }
        let err = parse_expr(&expr).expect_err("should fail with too deep");
        assert!(matches!(err.kind, crate::error::ParseErrorKind::TooDeep));
    }

    // Bit operations
    #[test]
    fn test_expr_bit_or() {
        let expr = parse("1 | 2");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::BitOr, _)));
    }

    #[test]
    fn test_expr_bit_and() {
        let expr = parse("1 & 2");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary(_, BinaryOp::BitAnd, _)
        ));
    }

    #[test]
    fn test_expr_bit_xor() {
        let expr = parse("1 ^ 2");
        assert!(matches!(
            expr.kind,
            ExprKind::Binary(_, BinaryOp::BitXor, _)
        ));
    }

    #[test]
    fn test_expr_shift_left() {
        let expr = parse("1 << 2");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Shl, _)));
    }

    #[test]
    fn test_expr_shift_right() {
        let expr = parse("1 >> 2");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Shr, _)));
    }

    #[test]
    fn test_expr_bit_not() {
        let expr = parse("~1");
        assert!(matches!(expr.kind, ExprKind::Unary(UnaryOp::BitNot, _)));
    }

    // NOT IN expression
    #[test]
    fn test_expr_not_in() {
        let expr = parse("x NOT IN (1, 2, 3)");
        if let ExprKind::In {
            list: InList::Values(values),
            negated,
            ..
        } = expr.kind
        {
            assert!(negated);
            assert_eq!(values.len(), 3);
        } else {
            panic!("expected IN expression");
        }
    }

    // from_lexer constructor
    #[test]
    fn test_from_lexer() {
        use crate::lexer::Lexer;
        let mut lexer = Lexer::new("1 + 2");
        let current = lexer.next_token();
        let mut parser = ExprParser::from_lexer(lexer, current);
        let expr = parser.parse_expr().unwrap();
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Add, _)));
    }

    // expect error case
    #[test]
    fn test_expect_wrong_token() {
        let mut parser = ExprParser::new("1 + 2");
        let result = parser.expect(&TokenKind::String(String::new()));
        assert!(result.is_err());
    }

    // EXISTS expression
    #[test]
    fn test_expr_exists() {
        let err = parse_err("EXISTS (SELECT 1)");
        // EXISTS returns an error since ExprParser doesn't have full SQL support
        assert!(err
            .to_string()
            .contains("EXISTS subqueries not yet implemented"));
    }

    // Empty tuple
    #[test]
    fn test_empty_tuple() {
        let expr = parse("()");
        if let ExprKind::Tuple(items) = expr.kind {
            assert!(items.is_empty());
        } else {
            panic!("expected tuple");
        }
    }

    // Division and modulo
    #[test]
    fn test_expr_division() {
        let expr = parse("a / b");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Div, _)));
    }

    #[test]
    fn test_expr_modulo() {
        let expr = parse("a % b");
        assert!(matches!(expr.kind, ExprKind::Binary(_, BinaryOp::Mod, _)));
    }

    // Aggregate functions
    #[test]
    fn test_expr_sum() {
        let expr = parse("SUM(x)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "SUM");
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_expr_avg() {
        let expr = parse("AVG(x)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "AVG");
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_expr_min() {
        let expr = parse("MIN(x)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "MIN");
        } else {
            panic!("expected function call");
        }
    }

    #[test]
    fn test_expr_max() {
        let expr = parse("MAX(x)");
        if let ExprKind::Call(call) = expr.kind {
            assert_eq!(call.name.name, "MAX");
        } else {
            panic!("expected function call");
        }
    }

    // Error: CASE with missing parts
    #[test]
    fn test_case_missing_then_error() {
        let err = parse_err("CASE WHEN x END");
        assert!(err.to_string().contains("THEN"));
    }

    // Error: qualified wildcard requires identifier
    #[test]
    fn test_qualified_wildcard_error() {
        let err = parse_err("(1 + 2).*");
        assert!(err.to_string().contains("identifier"));
    }

    // Test deeply nested expressions that might hit depth limit
    #[test]
    fn test_too_deep_expression() {
        // Create a deeply nested expression that exceeds MAX_DEPTH (64)
        let mut expr = "x".to_string();
        for _ in 0..70 {
            expr = format!("({})", expr);
        }
        let result = ExprParser::new(&expr).parse_expr();
        assert!(result.is_err());
    }
}
