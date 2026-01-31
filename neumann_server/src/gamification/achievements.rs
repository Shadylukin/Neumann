// SPDX-License-Identifier: MIT OR Apache-2.0
//! Achievement definitions and tracking.
//!
//! Achievements reward users for various milestones and encourage
//! exploration of all system features.

use serde::{Deserialize, Serialize};

/// Achievement tier determining rarity and XP reward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AchievementTier {
    /// Common achievements, easy to unlock.
    Bronze,
    /// Uncommon achievements requiring some effort.
    Silver,
    /// Rare achievements for dedicated users.
    Gold,
    /// Legendary achievements for mastery.
    Platinum,
}

impl AchievementTier {
    /// Returns the XP reward for unlocking an achievement of this tier.
    #[must_use]
    pub const fn xp_reward(&self) -> u64 {
        match self {
            Self::Bronze => 50,
            Self::Silver => 100,
            Self::Gold => 250,
            Self::Platinum => 500,
        }
    }

    /// Returns the CSS class for styling.
    #[must_use]
    pub const fn css_class(&self) -> &'static str {
        match self {
            Self::Bronze => "achievement-bronze",
            Self::Silver => "achievement-silver",
            Self::Gold => "achievement-gold",
            Self::Platinum => "achievement-platinum",
        }
    }

    /// Returns the display name.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Bronze => "Bronze",
            Self::Silver => "Silver",
            Self::Gold => "Gold",
            Self::Platinum => "Platinum",
        }
    }
}

/// Achievement category for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AchievementCategory {
    /// First-time discoveries and explorations.
    Discovery,
    /// Performance-related achievements.
    Performance,
    /// Consistency and dedication achievements.
    Dedication,
    /// Mastery of specific features.
    Mastery,
}

impl AchievementCategory {
    /// Returns the display name.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Discovery => "Discovery",
            Self::Performance => "Performance",
            Self::Dedication => "Dedication",
            Self::Mastery => "Mastery",
        }
    }

    /// Returns the icon character.
    #[must_use]
    pub const fn icon(&self) -> char {
        match self {
            Self::Discovery => '?',
            Self::Performance => '>',
            Self::Dedication => '*',
            Self::Mastery => '#',
        }
    }
}

/// An achievement definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Achievement {
    /// Unique identifier.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    /// Description of how to unlock.
    pub description: &'static str,
    /// Achievement tier.
    pub tier: AchievementTier,
    /// Achievement category.
    pub category: AchievementCategory,
    /// Threshold value for count-based achievements.
    pub threshold: Option<u64>,
    /// Whether this is a hidden/secret achievement.
    pub hidden: bool,
}

impl Achievement {
    /// Creates a new achievement.
    #[must_use]
    pub const fn new(
        id: &'static str,
        name: &'static str,
        description: &'static str,
        tier: AchievementTier,
        category: AchievementCategory,
    ) -> Self {
        Self {
            id,
            name,
            description,
            tier,
            category,
            threshold: None,
            hidden: false,
        }
    }

    /// Sets the threshold for count-based achievements.
    #[must_use]
    pub const fn with_threshold(mut self, threshold: u64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Marks as a hidden achievement.
    #[must_use]
    pub const fn hidden(mut self) -> Self {
        self.hidden = true;
        self
    }
}

/// All available achievements.
pub const ACHIEVEMENTS: &[Achievement] = &[
    // Discovery achievements
    Achievement::new(
        "first_query",
        "First Steps",
        "Execute your first query",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    ),
    Achievement::new(
        "first_graph",
        "Graph Explorer",
        "View the graph engine for the first time",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    ),
    Achievement::new(
        "first_vector",
        "Vector Pioneer",
        "Perform your first vector search",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    ),
    Achievement::new(
        "first_algorithm",
        "Algorithm Initiate",
        "Run a graph algorithm",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    ),
    Achievement::new(
        "all_engines",
        "Triple Threat",
        "Use all three engines in one session",
        AchievementTier::Silver,
        AchievementCategory::Discovery,
    ),
    // Performance achievements
    Achievement::new(
        "fast_query",
        "Speed Demon",
        "Execute a query in under 10ms",
        AchievementTier::Bronze,
        AchievementCategory::Performance,
    ),
    Achievement::new(
        "lightning_query",
        "Lightning Fast",
        "Execute a query in under 1ms",
        AchievementTier::Silver,
        AchievementCategory::Performance,
    ),
    Achievement::new(
        "hundred_queries",
        "Query Centurion",
        "Execute 100 queries",
        AchievementTier::Bronze,
        AchievementCategory::Performance,
    )
    .with_threshold(100),
    Achievement::new(
        "thousand_queries",
        "Query Master",
        "Execute 1,000 queries",
        AchievementTier::Silver,
        AchievementCategory::Performance,
    )
    .with_threshold(1000),
    Achievement::new(
        "ten_thousand_queries",
        "Query Legend",
        "Execute 10,000 queries",
        AchievementTier::Gold,
        AchievementCategory::Performance,
    )
    .with_threshold(10000),
    // Dedication achievements
    Achievement::new(
        "streak_3",
        "Getting Started",
        "Maintain a 3-day streak",
        AchievementTier::Bronze,
        AchievementCategory::Dedication,
    )
    .with_threshold(3),
    Achievement::new(
        "streak_7",
        "Weekly Warrior",
        "Maintain a 7-day streak",
        AchievementTier::Silver,
        AchievementCategory::Dedication,
    )
    .with_threshold(7),
    Achievement::new(
        "streak_30",
        "Monthly Maven",
        "Maintain a 30-day streak",
        AchievementTier::Gold,
        AchievementCategory::Dedication,
    )
    .with_threshold(30),
    Achievement::new(
        "streak_100",
        "Century Club",
        "Maintain a 100-day streak",
        AchievementTier::Platinum,
        AchievementCategory::Dedication,
    )
    .with_threshold(100),
    // Mastery achievements
    Achievement::new(
        "all_algorithms",
        "Algorithm Collector",
        "Use all graph algorithms",
        AchievementTier::Gold,
        AchievementCategory::Mastery,
    ),
    Achievement::new(
        "pagerank_master",
        "Influence Mapper",
        "Run PageRank 50 times",
        AchievementTier::Silver,
        AchievementCategory::Mastery,
    )
    .with_threshold(50),
    Achievement::new(
        "community_finder",
        "Community Builder",
        "Detect communities in 10 different graphs",
        AchievementTier::Silver,
        AchievementCategory::Mastery,
    )
    .with_threshold(10),
    Achievement::new(
        "path_finder",
        "Pathfinder",
        "Find 100 shortest paths",
        AchievementTier::Silver,
        AchievementCategory::Mastery,
    )
    .with_threshold(100),
    Achievement::new(
        "level_10",
        "Engine Master",
        "Reach level 10",
        AchievementTier::Gold,
        AchievementCategory::Mastery,
    )
    .with_threshold(10),
    Achievement::new(
        "level_20",
        "Tensor Grandmaster",
        "Reach level 20",
        AchievementTier::Platinum,
        AchievementCategory::Mastery,
    )
    .with_threshold(20),
    // Hidden achievements
    Achievement::new(
        "night_owl",
        "Night Owl",
        "Use the system between midnight and 4am",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    )
    .hidden(),
    Achievement::new(
        "early_bird",
        "Early Bird",
        "Use the system between 5am and 7am",
        AchievementTier::Bronze,
        AchievementCategory::Discovery,
    )
    .hidden(),
];

/// Get an achievement by ID.
#[must_use]
pub fn get_achievement(id: &str) -> Option<&'static Achievement> {
    ACHIEVEMENTS.iter().find(|a| a.id == id)
}

/// Get achievements by category.
#[must_use]
pub fn achievements_by_category(category: AchievementCategory) -> Vec<&'static Achievement> {
    ACHIEVEMENTS
        .iter()
        .filter(|a| a.category == category)
        .collect()
}

/// Get achievements by tier.
#[must_use]
pub fn achievements_by_tier(tier: AchievementTier) -> Vec<&'static Achievement> {
    ACHIEVEMENTS.iter().filter(|a| a.tier == tier).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tier_xp_reward() {
        assert_eq!(AchievementTier::Bronze.xp_reward(), 50);
        assert_eq!(AchievementTier::Silver.xp_reward(), 100);
        assert_eq!(AchievementTier::Gold.xp_reward(), 250);
        assert_eq!(AchievementTier::Platinum.xp_reward(), 500);
    }

    #[test]
    fn test_tier_css_class() {
        assert_eq!(AchievementTier::Bronze.css_class(), "achievement-bronze");
        assert_eq!(AchievementTier::Platinum.css_class(), "achievement-platinum");
    }

    #[test]
    fn test_category_display_name() {
        assert_eq!(AchievementCategory::Discovery.display_name(), "Discovery");
        assert_eq!(AchievementCategory::Mastery.display_name(), "Mastery");
    }

    #[test]
    fn test_category_icon() {
        assert_eq!(AchievementCategory::Discovery.icon(), '?');
        assert_eq!(AchievementCategory::Mastery.icon(), '#');
    }

    #[test]
    fn test_achievements_defined() {
        assert!(!ACHIEVEMENTS.is_empty());
        assert!(ACHIEVEMENTS.len() >= 20);
    }

    #[test]
    fn test_get_achievement() {
        let achievement = get_achievement("first_query");
        assert!(achievement.is_some());
        assert_eq!(achievement.unwrap().name, "First Steps");
    }

    #[test]
    fn test_get_achievement_not_found() {
        let achievement = get_achievement("nonexistent");
        assert!(achievement.is_none());
    }

    #[test]
    fn test_achievements_by_category() {
        let discovery = achievements_by_category(AchievementCategory::Discovery);
        assert!(!discovery.is_empty());
        assert!(discovery.iter().all(|a| a.category == AchievementCategory::Discovery));
    }

    #[test]
    fn test_achievements_by_tier() {
        let bronze = achievements_by_tier(AchievementTier::Bronze);
        assert!(!bronze.is_empty());
        assert!(bronze.iter().all(|a| a.tier == AchievementTier::Bronze));
    }

    #[test]
    fn test_achievement_with_threshold() {
        let achievement = get_achievement("hundred_queries").unwrap();
        assert_eq!(achievement.threshold, Some(100));
    }

    #[test]
    fn test_hidden_achievement() {
        let achievement = get_achievement("night_owl").unwrap();
        assert!(achievement.hidden);
    }

    #[test]
    fn test_achievement_serialization() {
        let achievement = get_achievement("first_query").unwrap();
        let json = serde_json::to_string(achievement).expect("serialization failed");
        assert!(json.contains("first_query"));
        assert!(json.contains("First Steps"));
    }
}
