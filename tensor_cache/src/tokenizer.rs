//! Token counting and cost estimation using tiktoken.

use crate::error::Result;
use std::sync::OnceLock;
use tiktoken_rs::CoreBPE;

/// Global encoder instance (lazy initialized).
static CL100K_ENCODER: OnceLock<CoreBPE> = OnceLock::new();

/// Token counter using tiktoken's cl100k_base encoding.
///
/// This is the encoding used by GPT-4, GPT-3.5-turbo, and text-embedding-ada-002.
pub struct TokenCounter;

impl TokenCounter {
    /// Get or initialize the encoder.
    fn encoder() -> Result<&'static CoreBPE> {
        // Use get_or_init with a panic fallback, then handle the result
        let encoder = CL100K_ENCODER.get_or_init(|| {
            tiktoken_rs::cl100k_base().expect("failed to load cl100k_base encoder")
        });
        Ok(encoder)
    }

    /// Count tokens in a text string.
    pub fn count(text: &str) -> Result<usize> {
        let encoder = Self::encoder()?;
        Ok(encoder.encode_ordinary(text).len())
    }

    /// Count tokens for a chat message (role + content + overhead).
    ///
    /// Each message has overhead tokens for formatting:
    /// - `<|im_start|>` role `\n` content `<|im_end|>`
    pub fn count_message(role: &str, content: &str) -> Result<usize> {
        let encoder = Self::encoder()?;
        let role_tokens = encoder.encode_ordinary(role).len();
        let content_tokens = encoder.encode_ordinary(content).len();
        // 4 tokens overhead per message: <|im_start|>, role, \n, <|im_end|>
        Ok(role_tokens + content_tokens + 4)
    }

    /// Count tokens for a list of chat messages.
    ///
    /// Includes per-message overhead and 3 tokens for assistant reply priming.
    pub fn count_messages(messages: &[(&str, &str)]) -> Result<usize> {
        let mut total = 0;
        for (role, content) in messages {
            total += Self::count_message(role, content)?;
        }
        // 3 tokens for <|im_start|>assistant prefix
        Ok(total + 3)
    }

    /// Estimate cost for a request in dollars.
    ///
    /// # Arguments
    /// * `input_tokens` - Number of input tokens
    /// * `output_tokens` - Number of output tokens
    /// * `input_rate` - Cost per 1000 input tokens
    /// * `output_rate` - Cost per 1000 output tokens
    pub fn estimate_cost(
        input_tokens: usize,
        output_tokens: usize,
        input_rate: f64,
        output_rate: f64,
    ) -> f64 {
        (input_tokens as f64 / 1000.0) * input_rate + (output_tokens as f64 / 1000.0) * output_rate
    }

    /// Estimate cost in micro-dollars (for integer storage).
    pub fn estimate_cost_microdollars(
        input_tokens: usize,
        output_tokens: usize,
        input_rate: f64,
        output_rate: f64,
    ) -> u64 {
        let dollars = Self::estimate_cost(input_tokens, output_tokens, input_rate, output_rate);
        (dollars * 1_000_000.0) as u64
    }
}

/// Model pricing information.
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Cost per 1000 input tokens in dollars.
    pub input_per_1k: f64,
    /// Cost per 1000 output tokens in dollars.
    pub output_per_1k: f64,
}

impl ModelPricing {
    /// GPT-4o pricing.
    pub const GPT4O: Self = Self {
        input_per_1k: 0.005,
        output_per_1k: 0.015,
    };

    /// GPT-4o mini pricing.
    pub const GPT4O_MINI: Self = Self {
        input_per_1k: 0.00015,
        output_per_1k: 0.0006,
    };

    /// GPT-4 Turbo pricing.
    pub const GPT4_TURBO: Self = Self {
        input_per_1k: 0.01,
        output_per_1k: 0.03,
    };

    /// GPT-3.5 Turbo pricing.
    pub const GPT35_TURBO: Self = Self {
        input_per_1k: 0.0005,
        output_per_1k: 0.0015,
    };

    /// Claude 3 Opus pricing.
    pub const CLAUDE3_OPUS: Self = Self {
        input_per_1k: 0.015,
        output_per_1k: 0.075,
    };

    /// Claude 3 Sonnet pricing.
    pub const CLAUDE3_SONNET: Self = Self {
        input_per_1k: 0.003,
        output_per_1k: 0.015,
    };

    /// Claude 3 Haiku pricing.
    pub const CLAUDE3_HAIKU: Self = Self {
        input_per_1k: 0.00025,
        output_per_1k: 0.00125,
    };

    /// Get pricing for a model by name.
    pub fn for_model(model: &str) -> Option<Self> {
        let model_lower = model.to_lowercase();
        if model_lower.contains("gpt-4o-mini") {
            Some(Self::GPT4O_MINI)
        } else if model_lower.contains("gpt-4o") {
            Some(Self::GPT4O)
        } else if model_lower.contains("gpt-4-turbo") {
            Some(Self::GPT4_TURBO)
        } else if model_lower.contains("gpt-3.5") {
            Some(Self::GPT35_TURBO)
        } else if model_lower.contains("claude-3-opus") || model_lower.contains("claude-opus") {
            Some(Self::CLAUDE3_OPUS)
        } else if model_lower.contains("claude-3-sonnet") || model_lower.contains("claude-sonnet") {
            Some(Self::CLAUDE3_SONNET)
        } else if model_lower.contains("claude-3-haiku") || model_lower.contains("claude-haiku") {
            Some(Self::CLAUDE3_HAIKU)
        } else {
            None
        }
    }

    /// Estimate cost for given tokens.
    pub fn estimate(&self, input_tokens: usize, output_tokens: usize) -> f64 {
        TokenCounter::estimate_cost(
            input_tokens,
            output_tokens,
            self.input_per_1k,
            self.output_per_1k,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_simple() {
        let count = TokenCounter::count("Hello, world!").unwrap();
        assert!(count > 0);
        assert!(count < 10); // Should be ~4 tokens
    }

    #[test]
    fn test_count_empty() {
        let count = TokenCounter::count("").unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_message() {
        let count = TokenCounter::count_message("user", "Hello!").unwrap();
        // "user" + "Hello!" + 4 overhead
        assert!(count > 4);
    }

    #[test]
    fn test_count_messages() {
        let messages = vec![("user", "Hello"), ("assistant", "Hi there!")];
        let count = TokenCounter::count_messages(&messages).unwrap();
        // Multiple messages + assistant priming
        assert!(count > 10);
    }

    #[test]
    fn test_estimate_cost() {
        // 1000 input tokens at $0.01/1k + 500 output tokens at $0.03/1k
        let cost = TokenCounter::estimate_cost(1000, 500, 0.01, 0.03);
        assert!((cost - 0.025).abs() < 0.0001);
    }

    #[test]
    fn test_estimate_cost_microdollars() {
        let micros = TokenCounter::estimate_cost_microdollars(1000, 500, 0.01, 0.03);
        assert_eq!(micros, 25000); // $0.025 = 25000 microdollars
    }

    #[test]
    fn test_model_pricing_lookup() {
        assert!(ModelPricing::for_model("gpt-4o").is_some());
        assert!(ModelPricing::for_model("gpt-4o-mini").is_some());
        assert!(ModelPricing::for_model("claude-3-opus").is_some());
        assert!(ModelPricing::for_model("unknown-model").is_none());
    }

    #[test]
    fn test_model_pricing_estimate() {
        let pricing = ModelPricing::GPT4_TURBO;
        let cost = pricing.estimate(1000, 1000);
        // $0.01 + $0.03 = $0.04
        assert!((cost - 0.04).abs() < 0.0001);
    }
}
