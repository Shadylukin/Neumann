use std::sync::OnceLock;

use tiktoken_rs::CoreBPE;

static CL100K_ENCODER: OnceLock<Option<CoreBPE>> = OnceLock::new();

/// Token counter using tiktoken's `cl100k_base` encoding (GPT-4, GPT-3.5-turbo, ada-002).
/// Falls back to character-based estimation (~4 chars per token) if tiktoken unavailable.
pub struct TokenCounter;

impl TokenCounter {
    fn encoder() -> Option<&'static CoreBPE> {
        CL100K_ENCODER
            .get_or_init(|| tiktoken_rs::cl100k_base().ok())
            .as_ref()
    }

    const fn estimate_tokens(text: &str) -> usize {
        // Fallback: ~4 characters per token for English text
        text.len().div_ceil(4)
    }

    #[must_use]
    pub fn count(text: &str) -> usize {
        Self::encoder().map_or_else(
            || Self::estimate_tokens(text),
            |enc| enc.encode_ordinary(text).len(),
        )
    }

    #[must_use]
    pub fn count_message(role: &str, content: &str) -> usize {
        Self::encoder().map_or_else(
            || Self::estimate_tokens(role) + Self::estimate_tokens(content) + 4,
            |enc| {
                let role_tokens = enc.encode_ordinary(role).len();
                let content_tokens = enc.encode_ordinary(content).len();
                role_tokens + content_tokens + 4
            },
        )
    }

    /// Count tokens for chat messages including 3 tokens for assistant reply priming.
    #[must_use]
    pub fn count_messages(messages: &[(&str, &str)]) -> usize {
        let mut total = 0;
        for (role, content) in messages {
            total += Self::count_message(role, content);
        }
        total + 3
    }

    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_cost(
        input_tokens: usize,
        output_tokens: usize,
        input_rate: f64,
        output_rate: f64,
    ) -> f64 {
        (input_tokens as f64 / 1000.0)
            .mul_add(input_rate, output_tokens as f64 / 1000.0 * output_rate)
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
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

/// Model pricing (cost per 1000 tokens in dollars).
#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
}

impl ModelPricing {
    pub const GPT4O: Self = Self {
        input_per_1k: 0.005,
        output_per_1k: 0.015,
    };

    pub const GPT4O_MINI: Self = Self {
        input_per_1k: 0.00015,
        output_per_1k: 0.0006,
    };

    pub const GPT4_TURBO: Self = Self {
        input_per_1k: 0.01,
        output_per_1k: 0.03,
    };

    pub const GPT35_TURBO: Self = Self {
        input_per_1k: 0.0005,
        output_per_1k: 0.0015,
    };

    pub const CLAUDE3_OPUS: Self = Self {
        input_per_1k: 0.015,
        output_per_1k: 0.075,
    };

    pub const CLAUDE3_SONNET: Self = Self {
        input_per_1k: 0.003,
        output_per_1k: 0.015,
    };

    pub const CLAUDE3_HAIKU: Self = Self {
        input_per_1k: 0.00025,
        output_per_1k: 0.00125,
    };

    #[must_use]
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

    #[must_use]
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
        let count = TokenCounter::count("Hello, world!");
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_count_empty() {
        let count = TokenCounter::count("");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_count_message() {
        let count = TokenCounter::count_message("user", "Hello!");
        assert!(count > 4);
    }

    #[test]
    fn test_count_messages() {
        let messages = vec![("user", "Hello"), ("assistant", "Hi there!")];
        let count = TokenCounter::count_messages(&messages);
        assert!(count > 10);
    }

    #[test]
    fn test_estimate_cost() {
        let cost = TokenCounter::estimate_cost(1000, 500, 0.01, 0.03);
        assert!((cost - 0.025).abs() < 0.0001);
    }

    #[test]
    fn test_estimate_cost_microdollars() {
        let micros = TokenCounter::estimate_cost_microdollars(1000, 500, 0.01, 0.03);
        assert_eq!(micros, 25000);
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
        assert!((cost - 0.04).abs() < 0.0001);
    }
}
