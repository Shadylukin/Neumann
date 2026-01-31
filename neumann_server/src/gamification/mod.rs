// SPDX-License-Identifier: MIT OR Apache-2.0
//! Gamification system for the Neumann admin UI.
//!
//! Provides XP/leveling, achievements, streaks, and progress tracking
//! to create an engaging, game-like experience.

mod achievements;
mod progress;

pub use achievements::{
    achievements_by_category, achievements_by_tier, get_achievement, Achievement,
    AchievementCategory, AchievementTier, ACHIEVEMENTS,
};
pub use progress::{DailyGoal, GoalType, UserProgress};

use serde::{Deserialize, Serialize};

/// XP requirements for each level.
const LEVEL_XP: [u64; 20] = [
    0,      // Level 1
    100,    // Level 2
    250,    // Level 3
    500,    // Level 4
    1000,   // Level 5
    1750,   // Level 6
    2750,   // Level 7
    4000,   // Level 8
    5500,   // Level 9
    7500,   // Level 10
    10000,  // Level 11
    13000,  // Level 12
    16500,  // Level 13
    20500,  // Level 14
    25000,  // Level 15
    30000,  // Level 16
    36000,  // Level 17
    43000,  // Level 18
    51000,  // Level 19
    60000,  // Level 20
];

/// Calculate level from total XP.
#[must_use]
pub fn level_from_xp(xp: u64) -> u32 {
    for (level, &required) in LEVEL_XP.iter().enumerate().rev() {
        if xp >= required {
            return (level + 1) as u32;
        }
    }
    1
}

/// Calculate XP progress within current level.
#[must_use]
pub fn level_progress(xp: u64) -> LevelProgress {
    let level = level_from_xp(xp);
    let level_idx = (level as usize).saturating_sub(1);

    let current_level_xp = LEVEL_XP.get(level_idx).copied().unwrap_or(0);
    let next_level_xp = LEVEL_XP.get(level_idx + 1).copied().unwrap_or(u64::MAX);

    let xp_in_level = xp.saturating_sub(current_level_xp);
    let xp_for_level = next_level_xp.saturating_sub(current_level_xp);

    let percentage = if xp_for_level > 0 {
        ((xp_in_level as f64 / xp_for_level as f64) * 100.0).min(100.0)
    } else {
        100.0
    };

    LevelProgress {
        level,
        current_xp: xp,
        xp_in_level,
        xp_for_level,
        percentage,
        is_max_level: level >= LEVEL_XP.len() as u32,
    }
}

/// Progress within current level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelProgress {
    /// Current level (1-20).
    pub level: u32,
    /// Total XP accumulated.
    pub current_xp: u64,
    /// XP earned within current level.
    pub xp_in_level: u64,
    /// XP required for next level.
    pub xp_for_level: u64,
    /// Percentage progress to next level.
    pub percentage: f64,
    /// Whether at max level.
    pub is_max_level: bool,
}

/// XP rewards for various actions.
#[derive(Debug, Clone, Copy)]
pub struct XpReward;

impl XpReward {
    /// XP for executing a query.
    pub const QUERY_EXECUTE: u64 = 1;
    /// XP for a fast query (under 10ms).
    pub const FAST_QUERY: u64 = 5;
    /// XP for running a graph algorithm.
    pub const ALGORITHM_RUN: u64 = 10;
    /// XP for first action of the day.
    pub const DAILY_FIRST: u64 = 25;
    /// XP for maintaining a streak.
    pub const STREAK_BONUS: u64 = 50;
    /// XP for unlocking an achievement.
    pub const ACHIEVEMENT_UNLOCK: u64 = 100;
}

/// Title earned at each level.
#[must_use]
pub fn level_title(level: u32) -> &'static str {
    match level {
        1 => "Novice Operator",
        2 => "Data Apprentice",
        3 => "Query Runner",
        4 => "Graph Walker",
        5 => "Vector Seeker",
        6 => "Index Builder",
        7 => "Schema Architect",
        8 => "Query Optimizer",
        9 => "Data Wrangler",
        10 => "Engine Master",
        11 => "Tensor Adept",
        12 => "Algorithm Sage",
        13 => "Performance Tuner",
        14 => "System Architect",
        15 => "Data Scientist",
        16 => "Graph Theorist",
        17 => "Vector Mathematician",
        18 => "Distributed Systems Expert",
        19 => "Neumann Veteran",
        20 => "Tensor Grandmaster",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_from_xp() {
        assert_eq!(level_from_xp(0), 1);
        assert_eq!(level_from_xp(50), 1);
        assert_eq!(level_from_xp(100), 2);
        assert_eq!(level_from_xp(250), 3);
        assert_eq!(level_from_xp(60000), 20);
        assert_eq!(level_from_xp(100000), 20);
    }

    #[test]
    fn test_level_progress() {
        let progress = level_progress(150);
        assert_eq!(progress.level, 2);
        assert_eq!(progress.xp_in_level, 50);
        assert_eq!(progress.xp_for_level, 150);
        assert!(!progress.is_max_level);
    }

    #[test]
    fn test_level_progress_max() {
        let progress = level_progress(60000);
        assert_eq!(progress.level, 20);
        assert!(progress.is_max_level);
    }

    #[test]
    fn test_level_progress_zero() {
        let progress = level_progress(0);
        assert_eq!(progress.level, 1);
        assert_eq!(progress.xp_in_level, 0);
        assert_eq!(progress.percentage, 0.0);
    }

    #[test]
    fn test_level_title() {
        assert_eq!(level_title(1), "Novice Operator");
        assert_eq!(level_title(10), "Engine Master");
        assert_eq!(level_title(20), "Tensor Grandmaster");
        assert_eq!(level_title(21), "Unknown");
    }

    #[test]
    fn test_xp_rewards() {
        assert_eq!(XpReward::QUERY_EXECUTE, 1);
        assert_eq!(XpReward::FAST_QUERY, 5);
        assert_eq!(XpReward::ACHIEVEMENT_UNLOCK, 100);
    }

    #[test]
    fn test_level_progress_serialization() {
        let progress = level_progress(500);
        let json = serde_json::to_string(&progress).expect("serialization failed");
        assert!(json.contains("level"));
        assert!(json.contains("current_xp"));
    }
}
