//! Lexer for the Neumann query language.
//!
//! Converts source text into a stream of tokens. Handles:
//! - Keywords (case-insensitive)
//! - Identifiers
//! - Numeric literals (integers and floats)
//! - String literals (single and double quoted, with escapes)
//! - Operators and punctuation
//! - Comments (-- and /* */)
//! - Whitespace (skipped)

use crate::span::{BytePos, Span};
use crate::token::{Token, TokenKind};
use std::str::Chars;

/// A lexer for tokenizing Neumann query language source.
pub struct Lexer<'a> {
    /// The source text being lexed.
    source: &'a str,
    /// Iterator over characters.
    chars: Chars<'a>,
    /// Current byte position.
    pos: u32,
    /// Peeked character (if any).
    peeked: Option<char>,
}

impl<'a> Lexer<'a> {
    /// Creates a new lexer for the given source.
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            chars: source.chars(),
            pos: 0,
            peeked: None,
        }
    }

    /// Returns the source text.
    pub fn source(&self) -> &'a str {
        self.source
    }

    /// Returns the current byte position.
    pub fn pos(&self) -> BytePos {
        BytePos(self.pos)
    }

    /// Peeks at the next character without consuming it.
    fn peek(&mut self) -> Option<char> {
        if self.peeked.is_none() {
            self.peeked = self.chars.next();
        }
        self.peeked
    }

    /// Peeks at the character after the next one.
    fn peek2(&self) -> Option<char> {
        let mut chars = self.chars.clone();
        if self.peeked.is_some() {
            chars.next()
        } else {
            chars.next();
            chars.next()
        }
    }

    /// Advances to the next character.
    fn advance(&mut self) -> Option<char> {
        let c = if let Some(c) = self.peeked.take() {
            c
        } else {
            self.chars.next()?
        };
        self.pos += c.len_utf8() as u32;
        Some(c)
    }

    /// Advances if the next character matches.
    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.advance();
            true
        } else {
            false
        }
    }

    /// Skips whitespace and comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(c) if c.is_whitespace() => {
                    self.advance();
                },
                Some('-') if self.peek2() == Some('-') => {
                    // Line comment: skip to end of line
                    self.advance();
                    self.advance();
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                },
                Some('/') if self.peek2() == Some('*') => {
                    // Block comment: skip to */
                    self.advance();
                    self.advance();
                    let mut depth = 1;
                    while depth > 0 {
                        match self.peek() {
                            Some('/') if self.peek2() == Some('*') => {
                                self.advance();
                                self.advance();
                                depth += 1;
                            },
                            Some('*') if self.peek2() == Some('/') => {
                                self.advance();
                                self.advance();
                                depth -= 1;
                            },
                            Some(_) => {
                                self.advance();
                            },
                            None => break,
                        }
                    }
                },
                _ => break,
            }
        }
    }

    /// Scans an identifier or keyword.
    fn scan_ident(&mut self, start: u32) -> Token {
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text = &self.source[start as usize..self.pos as usize];
        let span = Span::from_offsets(start, self.pos);

        let kind =
            TokenKind::keyword_from_str(text).unwrap_or_else(|| TokenKind::Ident(text.to_string()));

        Token::new(kind, span)
    }

    /// Scans a numeric literal (integer or float).
    fn scan_number(&mut self, start: u32) -> Token {
        // Consume digits
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }

        // Check for decimal point
        let is_float =
            if self.peek() == Some('.') && self.peek2().is_some_and(|c| c.is_ascii_digit()) {
                self.advance(); // consume '.'
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        self.advance();
                    } else {
                        break;
                    }
                }
                true
            } else {
                false
            };

        // Check for exponent
        let has_exponent = if let Some('e' | 'E') = self.peek() {
            self.advance();
            if let Some('+' | '-') = self.peek() {
                self.advance();
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    self.advance();
                } else {
                    break;
                }
            }
            true
        } else {
            false
        };

        let text = &self.source[start as usize..self.pos as usize];
        let span = Span::from_offsets(start, self.pos);

        let kind = if is_float || has_exponent {
            match text.parse::<f64>() {
                Ok(n) => TokenKind::Float(n),
                Err(e) => TokenKind::Error(format!("invalid float: {}", e)),
            }
        } else {
            match text.parse::<i64>() {
                Ok(n) => TokenKind::Integer(n),
                Err(e) => TokenKind::Error(format!("invalid integer: {}", e)),
            }
        };

        Token::new(kind, span)
    }

    /// Scans a string literal.
    fn scan_string(&mut self, start: u32, quote: char) -> Token {
        let mut value = String::new();
        let mut terminated = false;

        loop {
            match self.peek() {
                Some(c) if c == quote => {
                    self.advance();
                    // Check for escaped quote (doubled quote)
                    if self.peek() == Some(quote) {
                        self.advance();
                        value.push(quote);
                    } else {
                        terminated = true;
                        break;
                    }
                },
                Some('\\') => {
                    self.advance();
                    match self.peek() {
                        Some('n') => {
                            self.advance();
                            value.push('\n');
                        },
                        Some('r') => {
                            self.advance();
                            value.push('\r');
                        },
                        Some('t') => {
                            self.advance();
                            value.push('\t');
                        },
                        Some('\\') => {
                            self.advance();
                            value.push('\\');
                        },
                        Some('\'') => {
                            self.advance();
                            value.push('\'');
                        },
                        Some('"') => {
                            self.advance();
                            value.push('"');
                        },
                        Some('0') => {
                            self.advance();
                            value.push('\0');
                        },
                        Some(c) => {
                            self.advance();
                            value.push('\\');
                            value.push(c);
                        },
                        None => {
                            value.push('\\');
                        },
                    }
                },
                Some('\n') | None => {
                    break;
                },
                Some(c) => {
                    self.advance();
                    value.push(c);
                },
            }
        }

        let span = Span::from_offsets(start, self.pos);

        let kind = if terminated {
            TokenKind::String(value)
        } else {
            TokenKind::Error("unterminated string literal".to_string())
        };

        Token::new(kind, span)
    }

    /// Scans the next token.
    pub fn next_token(&mut self) -> Token {
        self.skip_whitespace_and_comments();

        let start = self.pos;

        let c = match self.advance() {
            Some(c) => c,
            None => {
                return Token::new(TokenKind::Eof, Span::point(BytePos(start)));
            },
        };

        match c {
            // Identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => self.scan_ident(start),

            // Numbers
            '0'..='9' => self.scan_number(start),

            // Strings
            '\'' | '"' => self.scan_string(start, c),

            // Operators and punctuation
            '+' => Token::new(TokenKind::Plus, Span::from_offsets(start, self.pos)),
            '-' => {
                if self.eat('>') {
                    Token::new(TokenKind::Arrow, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Minus, Span::from_offsets(start, self.pos))
                }
            },
            '*' => Token::new(TokenKind::Star, Span::from_offsets(start, self.pos)),
            '/' => Token::new(TokenKind::Slash, Span::from_offsets(start, self.pos)),
            '%' => Token::new(TokenKind::Percent, Span::from_offsets(start, self.pos)),
            '=' => {
                if self.eat('>') {
                    Token::new(TokenKind::FatArrow, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Eq, Span::from_offsets(start, self.pos))
                }
            },
            '!' => {
                if self.eat('=') {
                    Token::new(TokenKind::Ne, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Bang, Span::from_offsets(start, self.pos))
                }
            },
            '<' => {
                if self.eat('=') {
                    Token::new(TokenKind::Le, Span::from_offsets(start, self.pos))
                } else if self.eat('>') {
                    Token::new(TokenKind::Ne, Span::from_offsets(start, self.pos))
                } else if self.eat('<') {
                    Token::new(TokenKind::Shl, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Lt, Span::from_offsets(start, self.pos))
                }
            },
            '>' => {
                if self.eat('=') {
                    Token::new(TokenKind::Ge, Span::from_offsets(start, self.pos))
                } else if self.eat('>') {
                    Token::new(TokenKind::Shr, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Gt, Span::from_offsets(start, self.pos))
                }
            },
            '&' => {
                if self.eat('&') {
                    Token::new(TokenKind::AmpAmp, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Amp, Span::from_offsets(start, self.pos))
                }
            },
            '|' => {
                if self.eat('|') {
                    Token::new(TokenKind::Concat, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Pipe, Span::from_offsets(start, self.pos))
                }
            },
            '^' => Token::new(TokenKind::Caret, Span::from_offsets(start, self.pos)),
            '~' => Token::new(TokenKind::Tilde, Span::from_offsets(start, self.pos)),
            '(' => Token::new(TokenKind::LParen, Span::from_offsets(start, self.pos)),
            ')' => Token::new(TokenKind::RParen, Span::from_offsets(start, self.pos)),
            '[' => Token::new(TokenKind::LBracket, Span::from_offsets(start, self.pos)),
            ']' => Token::new(TokenKind::RBracket, Span::from_offsets(start, self.pos)),
            '{' => Token::new(TokenKind::LBrace, Span::from_offsets(start, self.pos)),
            '}' => Token::new(TokenKind::RBrace, Span::from_offsets(start, self.pos)),
            ',' => Token::new(TokenKind::Comma, Span::from_offsets(start, self.pos)),
            '.' => Token::new(TokenKind::Dot, Span::from_offsets(start, self.pos)),
            ';' => Token::new(TokenKind::Semicolon, Span::from_offsets(start, self.pos)),
            ':' => {
                if self.eat(':') {
                    Token::new(TokenKind::ColonColon, Span::from_offsets(start, self.pos))
                } else {
                    Token::new(TokenKind::Colon, Span::from_offsets(start, self.pos))
                }
            },
            '?' => Token::new(TokenKind::Question, Span::from_offsets(start, self.pos)),
            '@' => Token::new(TokenKind::At, Span::from_offsets(start, self.pos)),
            '#' => Token::new(TokenKind::Hash, Span::from_offsets(start, self.pos)),
            '$' => Token::new(TokenKind::Dollar, Span::from_offsets(start, self.pos)),

            _ => Token::new(
                TokenKind::Error(format!("unexpected character: '{}'", c)),
                Span::from_offsets(start, self.pos),
            ),
        }
    }

    /// Tokenizes the entire source, returning all tokens.
    pub fn tokenize(mut self) -> Vec<Token> {
        let mut tokens = Vec::new();
        loop {
            let token = self.next_token();
            let is_eof = token.is_eof();
            tokens.push(token);
            if is_eof {
                break;
            }
        }
        tokens
    }
}

/// Tokenizes source text into a vector of tokens.
pub fn tokenize(source: &str) -> Vec<Token> {
    Lexer::new(source).tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> Vec<TokenKind> {
        tokenize(source).into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_empty() {
        assert_eq!(tokens(""), vec![TokenKind::Eof]);
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(tokens("   \n\t  "), vec![TokenKind::Eof]);
    }

    #[test]
    fn test_keywords() {
        assert_eq!(
            tokens("SELECT FROM WHERE"),
            vec![
                TokenKind::Select,
                TokenKind::From,
                TokenKind::Where,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_case_insensitive_keywords() {
        assert_eq!(
            tokens("select FROM wHeRe"),
            vec![
                TokenKind::Select,
                TokenKind::From,
                TokenKind::Where,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_identifiers() {
        assert_eq!(
            tokens("users user_id _private"),
            vec![
                TokenKind::Ident("users".to_string()),
                TokenKind::Ident("user_id".to_string()),
                TokenKind::Ident("_private".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_integers() {
        assert_eq!(
            tokens("0 42 12345"),
            vec![
                TokenKind::Integer(0),
                TokenKind::Integer(42),
                TokenKind::Integer(12345),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_floats() {
        assert_eq!(
            tokens("3.14 0.5 10.0"),
            vec![
                TokenKind::Float(3.14),
                TokenKind::Float(0.5),
                TokenKind::Float(10.0),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_scientific_notation() {
        assert_eq!(
            tokens("1e10 2.5E-3 1e+5"),
            vec![
                TokenKind::Float(1e10),
                TokenKind::Float(2.5e-3),
                TokenKind::Float(1e+5),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_single_quoted_strings() {
        assert_eq!(
            tokens("'hello' 'world'"),
            vec![
                TokenKind::String("hello".to_string()),
                TokenKind::String("world".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_double_quoted_strings() {
        assert_eq!(
            tokens("\"hello\" \"world\""),
            vec![
                TokenKind::String("hello".to_string()),
                TokenKind::String("world".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_string_escapes() {
        assert_eq!(
            tokens(r"'hello\nworld' 'tab\there'"),
            vec![
                TokenKind::String("hello\nworld".to_string()),
                TokenKind::String("tab\there".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_doubled_quote_escape() {
        assert_eq!(
            tokens("'it''s'"),
            vec![TokenKind::String("it's".to_string()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_operators() {
        assert_eq!(
            tokens("+ - * / %"),
            vec![
                TokenKind::Plus,
                TokenKind::Minus,
                TokenKind::Star,
                TokenKind::Slash,
                TokenKind::Percent,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_comparison_operators() {
        assert_eq!(
            tokens("= != <> < <= > >="),
            vec![
                TokenKind::Eq,
                TokenKind::Ne,
                TokenKind::Ne,
                TokenKind::Lt,
                TokenKind::Le,
                TokenKind::Gt,
                TokenKind::Ge,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_logical_operators() {
        assert_eq!(
            tokens("&& || !"),
            vec![
                TokenKind::AmpAmp,
                TokenKind::Concat,
                TokenKind::Bang,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_punctuation() {
        assert_eq!(
            tokens("( ) [ ] { } , . ; :"),
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Comma,
                TokenKind::Dot,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_arrows() {
        assert_eq!(
            tokens("-> => ::"),
            vec![
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::ColonColon,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_bitwise_operators() {
        assert_eq!(
            tokens("& | ^ ~ << >>"),
            vec![
                TokenKind::Amp,
                TokenKind::Pipe,
                TokenKind::Caret,
                TokenKind::Tilde,
                TokenKind::Shl,
                TokenKind::Shr,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_special_chars() {
        assert_eq!(
            tokens("? @ # $"),
            vec![
                TokenKind::Question,
                TokenKind::At,
                TokenKind::Hash,
                TokenKind::Dollar,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_line_comment() {
        assert_eq!(
            tokens("SELECT -- this is a comment\nFROM"),
            vec![TokenKind::Select, TokenKind::From, TokenKind::Eof]
        );
    }

    #[test]
    fn test_block_comment() {
        assert_eq!(
            tokens("SELECT /* comment */ FROM"),
            vec![TokenKind::Select, TokenKind::From, TokenKind::Eof]
        );
    }

    #[test]
    fn test_nested_block_comment() {
        assert_eq!(
            tokens("SELECT /* outer /* inner */ still comment */ FROM"),
            vec![TokenKind::Select, TokenKind::From, TokenKind::Eof]
        );
    }

    #[test]
    fn test_simple_query() {
        let source = "SELECT * FROM users WHERE id = 1";
        assert_eq!(
            tokens(source),
            vec![
                TokenKind::Select,
                TokenKind::Star,
                TokenKind::From,
                TokenKind::Ident("users".to_string()),
                TokenKind::Where,
                TokenKind::Ident("id".to_string()),
                TokenKind::Eq,
                TokenKind::Integer(1),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_graph_query() {
        let source = "NODE CREATE user {name: 'Alice'}";
        assert_eq!(
            tokens(source),
            vec![
                TokenKind::Node,
                TokenKind::Create,
                TokenKind::Ident("user".to_string()),
                TokenKind::LBrace,
                TokenKind::Ident("name".to_string()),
                TokenKind::Colon,
                TokenKind::String("Alice".to_string()),
                TokenKind::RBrace,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_vector_query() {
        let source = "SIMILAR 'query' LIMIT 10";
        assert_eq!(
            tokens(source),
            vec![
                TokenKind::Similar,
                TokenKind::String("query".to_string()),
                TokenKind::Limit,
                TokenKind::Integer(10),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_spans() {
        let source = "SELECT FROM";
        let tokens = tokenize(source);

        assert_eq!(tokens[0].span, Span::from_offsets(0, 6));
        assert_eq!(tokens[0].span.extract(source), "SELECT");

        assert_eq!(tokens[1].span, Span::from_offsets(7, 11));
        assert_eq!(tokens[1].span.extract(source), "FROM");
    }

    #[test]
    fn test_lexer_pos() {
        let mut lexer = Lexer::new("SELECT");
        assert_eq!(lexer.pos(), BytePos(0));
        lexer.next_token();
        assert_eq!(lexer.pos(), BytePos(6));
    }

    #[test]
    fn test_lexer_source() {
        let source = "SELECT * FROM users";
        let lexer = Lexer::new(source);
        assert_eq!(lexer.source(), source);
    }

    #[test]
    fn test_unterminated_string() {
        let tokens = tokenize("'unterminated");
        assert!(matches!(
            &tokens[0].kind,
            TokenKind::Error(msg) if msg.contains("unterminated")
        ));
    }

    #[test]
    fn test_unexpected_character() {
        let tokens = tokenize("SELECT ` FROM");
        assert!(matches!(
            &tokens[1].kind,
            TokenKind::Error(msg) if msg.contains("unexpected")
        ));
    }

    #[test]
    fn test_dot_without_decimal() {
        // "3." should be parsed as integer 3, then dot
        assert_eq!(
            tokens("3. "),
            vec![TokenKind::Integer(3), TokenKind::Dot, TokenKind::Eof]
        );
    }

    #[test]
    fn test_all_escape_sequences() {
        // Test escape sequences: \n \r \t
        assert_eq!(
            tokens("'\\n\\r\\t'"),
            vec![TokenKind::String("\n\r\t".to_string()), TokenKind::Eof]
        );
        // Test escape sequences: \\ \"
        assert_eq!(
            tokens("'\\\\\\\"'"),
            vec![TokenKind::String("\\\"".to_string()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_unknown_escape() {
        // Unknown escapes pass through
        assert_eq!(
            tokens(r"'\x'"),
            vec![TokenKind::String("\\x".to_string()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_multiline_block_comment() {
        let source = "SELECT\n/* multi\nline\ncomment */\nFROM";
        assert_eq!(
            tokens(source),
            vec![TokenKind::Select, TokenKind::From, TokenKind::Eof]
        );
    }

    #[test]
    fn test_adjacent_tokens() {
        assert_eq!(
            tokens("1+2*3"),
            vec![
                TokenKind::Integer(1),
                TokenKind::Plus,
                TokenKind::Integer(2),
                TokenKind::Star,
                TokenKind::Integer(3),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_identifier_with_numbers() {
        assert_eq!(
            tokens("user123 _id1"),
            vec![
                TokenKind::Ident("user123".to_string()),
                TokenKind::Ident("_id1".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_all_graph_keywords() {
        assert_eq!(
            tokens("NODE EDGE NEIGHBORS PATH OUTGOING INCOMING BOTH SHORTEST"),
            vec![
                TokenKind::Node,
                TokenKind::Edge,
                TokenKind::Neighbors,
                TokenKind::Path,
                TokenKind::Outgoing,
                TokenKind::Incoming,
                TokenKind::Both,
                TokenKind::Shortest,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_all_vector_keywords() {
        assert_eq!(
            tokens("EMBED SIMILAR VECTOR COSINE EUCLIDEAN DOT_PRODUCT"),
            vec![
                TokenKind::Embed,
                TokenKind::Similar,
                TokenKind::Vector,
                TokenKind::Cosine,
                TokenKind::Euclidean,
                TokenKind::DotProduct,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_complex_query() {
        let source = r#"
            SELECT u.name, COUNT(*) as count
            FROM users u
            JOIN orders o ON u.id = o.user_id
            WHERE u.active = TRUE AND o.total >= 100.50
            GROUP BY u.name
            HAVING COUNT(*) > 5
            ORDER BY count DESC
            LIMIT 10
        "#;

        let toks = tokens(source);
        assert!(toks.contains(&TokenKind::Select));
        assert!(toks.contains(&TokenKind::From));
        assert!(toks.contains(&TokenKind::Join));
        assert!(toks.contains(&TokenKind::Where));
        assert!(toks.contains(&TokenKind::Group));
        assert!(toks.contains(&TokenKind::Having));
        assert!(toks.contains(&TokenKind::Order));
        assert!(toks.contains(&TokenKind::Limit));
        assert!(toks.contains(&TokenKind::True));
        assert!(toks.contains(&TokenKind::Count));
    }
}
