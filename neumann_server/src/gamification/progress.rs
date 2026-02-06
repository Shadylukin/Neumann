// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! User progress tracking for gamification.
//!
//! Tracks XP, achievements, streaks, and daily goals.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use super::achievements::AchievementTier;
use super::{level_from_xp, level_progress, LevelProgress};

/// User's gamification progress.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProgress {
    /// Total XP accumulated.
    pub xp_total: u64,
    /// IDs of unlocked achievements.
    pub achievements_unlocked: HashSet<String>,
    /// Current streak in days.
    pub streak_current: u32,
    /// Longest streak ever achieved.
    pub streak_best: u32,
    /// Last activity date (YYYYMMDD format).
    pub last_activity_date: u32,
    /// Daily goals and their progress.
    pub daily_goals: Vec<DailyGoal>,
    /// Statistics counters.
    pub stats: UserStats,
    /// Algorithms used (for "all algorithms" achievement).
    pub algorithms_used: HashSet<String>,
}

impl UserProgress {
    /// Creates new empty progress.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the current level.
    #[must_use]
    pub fn level(&self) -> u32 {
        level_from_xp(self.xp_total)
    }

    /// Gets detailed level progress.
    #[must_use]
    pub fn level_progress(&self) -> LevelProgress {
        level_progress(self.xp_total)
    }

    /// Awards XP and returns the new total.
    pub fn award_xp(&mut self, amount: u64) -> u64 {
        self.xp_total = self.xp_total.saturating_add(amount);
        self.xp_total
    }

    /// Unlocks an achievement and returns the XP reward.
    pub fn unlock_achievement(&mut self, id: &str, tier: AchievementTier) -> u64 {
        if self.achievements_unlocked.insert(id.to_string()) {
            let reward = tier.xp_reward();
            self.award_xp(reward);
            reward
        } else {
            0
        }
    }

    /// Checks if an achievement is unlocked.
    #[must_use]
    pub fn has_achievement(&self, id: &str) -> bool {
        self.achievements_unlocked.contains(id)
    }

    /// Gets the count of unlocked achievements.
    #[must_use]
    pub fn achievement_count(&self) -> usize {
        self.achievements_unlocked.len()
    }

    /// Updates the streak based on current date.
    pub fn update_streak(&mut self, today: u32) {
        if self.last_activity_date == 0 {
            // First activity ever
            self.streak_current = 1;
            self.streak_best = 1;
        } else if today == self.last_activity_date {
            // Same day, no change
        } else if today == self.last_activity_date + 1
            || (today % 100 == 1 && self.is_consecutive_month(today))
        {
            // Consecutive day
            self.streak_current += 1;
            self.streak_best = self.streak_best.max(self.streak_current);
        } else {
            // Streak broken
            self.streak_current = 1;
        }
        self.last_activity_date = today;
    }

    /// Checks if the date is the first of a consecutive month.
    fn is_consecutive_month(&self, today: u32) -> bool {
        let last_year = self.last_activity_date / 10000;
        let last_month = (self.last_activity_date / 100) % 100;
        let last_day = self.last_activity_date % 100;

        let today_year = today / 10000;
        let today_month = (today / 100) % 100;

        // Check if today is the 1st and last was the last day of previous month
        if today % 100 != 1 {
            return false;
        }

        // Simple check: same year, consecutive months, last day was >= 28
        if today_year == last_year && today_month == last_month + 1 && last_day >= 28 {
            return true;
        }

        // Year rollover: December -> January
        if today_year == last_year + 1 && today_month == 1 && last_month == 12 && last_day == 31 {
            return true;
        }

        false
    }

    /// Records an algorithm usage.
    pub fn record_algorithm(&mut self, algorithm: &str) {
        self.algorithms_used.insert(algorithm.to_string());
    }

    /// Gets the count of unique algorithms used.
    #[must_use]
    pub fn algorithms_used_count(&self) -> usize {
        self.algorithms_used.len()
    }

    /// Gets daily goal progress percentage.
    #[must_use]
    pub fn daily_goal_progress(&self) -> f64 {
        if self.daily_goals.is_empty() {
            return 0.0;
        }

        let completed = self.daily_goals.iter().filter(|g| g.is_complete()).count();
        #[allow(clippy::cast_precision_loss)]
        let pct = (completed as f64 / self.daily_goals.len() as f64) * 100.0;
        pct
    }

    /// Resets daily goals for a new day.
    pub fn reset_daily_goals(&mut self) {
        self.daily_goals = vec![
            DailyGoal::new(GoalType::Queries, 10),
            DailyGoal::new(GoalType::Algorithms, 3),
            DailyGoal::new(GoalType::FastQueries, 5),
        ];
    }
}

/// Statistics counters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserStats {
    /// Total queries executed.
    pub queries_total: u64,
    /// Fast queries (under 10ms).
    pub queries_fast: u64,
    /// Lightning queries (under 1ms).
    pub queries_lightning: u64,
    /// Graph algorithm runs.
    pub algorithms_run: u64,
    /// Vector searches performed.
    pub vector_searches: u64,
    /// Graph traversals performed.
    pub graph_traversals: u64,
    /// Shortest paths found.
    pub paths_found: u64,
    /// Communities detected.
    pub communities_detected: u64,
}

impl UserStats {
    /// Increments query count.
    pub fn record_query(&mut self, latency_ms: f64) {
        self.queries_total += 1;
        if latency_ms < 10.0 {
            self.queries_fast += 1;
        }
        if latency_ms < 1.0 {
            self.queries_lightning += 1;
        }
    }

    /// Increments algorithm run count.
    pub fn record_algorithm(&mut self) {
        self.algorithms_run += 1;
    }

    /// Increments vector search count.
    pub fn record_vector_search(&mut self) {
        self.vector_searches += 1;
    }

    /// Increments graph traversal count.
    pub fn record_graph_traversal(&mut self) {
        self.graph_traversals += 1;
    }

    /// Increments path found count.
    pub fn record_path_found(&mut self) {
        self.paths_found += 1;
    }

    /// Increments community detection count.
    pub fn record_community_detected(&mut self) {
        self.communities_detected += 1;
    }
}

/// A daily goal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyGoal {
    /// Type of goal.
    pub goal_type: GoalType,
    /// Target value to reach.
    pub target: u32,
    /// Current progress.
    pub current: u32,
}

impl DailyGoal {
    /// Creates a new daily goal.
    #[must_use]
    pub const fn new(goal_type: GoalType, target: u32) -> Self {
        Self {
            goal_type,
            target,
            current: 0,
        }
    }

    /// Checks if the goal is complete.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.current >= self.target
    }

    /// Gets progress as a percentage.
    #[must_use]
    pub fn progress_percent(&self) -> f64 {
        if self.target == 0 {
            return 100.0;
        }
        ((f64::from(self.current) / f64::from(self.target)) * 100.0).min(100.0)
    }

    /// Increments progress.
    pub fn increment(&mut self) {
        self.current = self.current.saturating_add(1);
    }
}

/// Types of daily goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalType {
    /// Execute N queries.
    Queries,
    /// Run N algorithms.
    Algorithms,
    /// Execute N fast queries.
    FastQueries,
    /// Perform N vector searches.
    VectorSearches,
    /// Find N paths.
    PathsFound,
}

impl GoalType {
    /// Returns the display name.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Queries => "Execute Queries",
            Self::Algorithms => "Run Algorithms",
            Self::FastQueries => "Fast Queries",
            Self::VectorSearches => "Vector Searches",
            Self::PathsFound => "Find Paths",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_progress_new() {
        let progress = UserProgress::new();
        assert_eq!(progress.xp_total, 0);
        assert_eq!(progress.level(), 1);
        assert!(progress.achievements_unlocked.is_empty());
    }

    #[test]
    fn test_award_xp() {
        let mut progress = UserProgress::new();
        progress.award_xp(100);
        assert_eq!(progress.xp_total, 100);
        assert_eq!(progress.level(), 2);
    }

    #[test]
    fn test_unlock_achievement() {
        let mut progress = UserProgress::new();
        let reward = progress.unlock_achievement("first_query", AchievementTier::Bronze);
        assert_eq!(reward, 50);
        assert!(progress.has_achievement("first_query"));
        assert_eq!(progress.xp_total, 50);
    }

    #[test]
    fn test_unlock_achievement_duplicate() {
        let mut progress = UserProgress::new();
        progress.unlock_achievement("first_query", AchievementTier::Bronze);
        let reward = progress.unlock_achievement("first_query", AchievementTier::Bronze);
        assert_eq!(reward, 0);
        assert_eq!(progress.xp_total, 50);
    }

    #[test]
    fn test_streak_first_activity() {
        let mut progress = UserProgress::new();
        progress.update_streak(2024_01_15);
        assert_eq!(progress.streak_current, 1);
        assert_eq!(progress.streak_best, 1);
    }

    #[test]
    fn test_streak_consecutive() {
        let mut progress = UserProgress::new();
        progress.update_streak(2024_01_15);
        progress.update_streak(2024_01_16);
        progress.update_streak(2024_01_17);
        assert_eq!(progress.streak_current, 3);
        assert_eq!(progress.streak_best, 3);
    }

    #[test]
    fn test_streak_broken() {
        let mut progress = UserProgress::new();
        progress.update_streak(2024_01_15);
        progress.update_streak(2024_01_16);
        progress.update_streak(2024_01_18); // Skipped a day
        assert_eq!(progress.streak_current, 1);
        assert_eq!(progress.streak_best, 2);
    }

    #[test]
    fn test_streak_same_day() {
        let mut progress = UserProgress::new();
        progress.update_streak(2024_01_15);
        progress.update_streak(2024_01_15);
        assert_eq!(progress.streak_current, 1);
    }

    #[test]
    fn test_record_algorithm() {
        let mut progress = UserProgress::new();
        progress.record_algorithm("pagerank");
        progress.record_algorithm("louvain");
        progress.record_algorithm("pagerank"); // Duplicate
        assert_eq!(progress.algorithms_used_count(), 2);
    }

    #[test]
    fn test_daily_goal_progress() {
        let mut goal = DailyGoal::new(GoalType::Queries, 10);
        assert!(!goal.is_complete());
        assert_eq!(goal.progress_percent(), 0.0);

        goal.increment();
        goal.increment();
        goal.increment();
        assert!(!goal.is_complete());
        assert_eq!(goal.progress_percent(), 30.0);

        for _ in 0..7 {
            goal.increment();
        }
        assert!(goal.is_complete());
        assert_eq!(goal.progress_percent(), 100.0);
    }

    #[test]
    fn test_reset_daily_goals() {
        let mut progress = UserProgress::new();
        progress.reset_daily_goals();
        assert_eq!(progress.daily_goals.len(), 3);
    }

    #[test]
    fn test_user_stats_record_query() {
        let mut stats = UserStats::default();
        stats.record_query(5.0);
        assert_eq!(stats.queries_total, 1);
        assert_eq!(stats.queries_fast, 1);
        assert_eq!(stats.queries_lightning, 0);

        stats.record_query(0.5);
        assert_eq!(stats.queries_total, 2);
        assert_eq!(stats.queries_fast, 2);
        assert_eq!(stats.queries_lightning, 1);
    }

    #[test]
    fn test_goal_type_display_name() {
        assert_eq!(GoalType::Queries.display_name(), "Execute Queries");
        assert_eq!(GoalType::Algorithms.display_name(), "Run Algorithms");
    }

    #[test]
    fn test_user_progress_serialization() {
        let mut progress = UserProgress::new();
        progress.award_xp(500);
        progress.unlock_achievement("first_query", AchievementTier::Bronze);

        let json = serde_json::to_string(&progress).expect("serialization failed");
        assert!(json.contains("xp_total"));
        assert!(json.contains("achievements_unlocked"));

        let decoded: UserProgress = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.xp_total, 550); // 500 + 50 from achievement
    }

    // ========== is_consecutive_month tests ==========

    #[test]
    fn test_streak_consecutive_month_rollover() {
        let mut progress = UserProgress::new();
        // January 31
        progress.update_streak(20240131);
        // February 1 (should continue streak)
        progress.update_streak(20240201);
        assert_eq!(progress.streak_current, 2);
    }

    #[test]
    fn test_streak_year_rollover() {
        let mut progress = UserProgress::new();
        // December 31
        progress.update_streak(20231231);
        // January 1 (should continue streak)
        progress.update_streak(20240101);
        assert_eq!(progress.streak_current, 2);
    }

    #[test]
    fn test_streak_not_consecutive_month() {
        let mut progress = UserProgress::new();
        // January 15
        progress.update_streak(20240115);
        // February 1 (not consecutive, skipped days)
        progress.update_streak(20240201);
        assert_eq!(progress.streak_current, 1);
    }

    #[test]
    fn test_streak_not_first_of_month() {
        let mut progress = UserProgress::new();
        // January 30
        progress.update_streak(20240130);
        // February 2 (not the 1st)
        progress.update_streak(20240202);
        assert_eq!(progress.streak_current, 1);
    }

    // ========== daily_goal_progress tests ==========

    #[test]
    fn test_daily_goal_progress_empty() {
        let progress = UserProgress::new();
        assert_eq!(progress.daily_goal_progress(), 0.0);
    }

    #[test]
    fn test_daily_goal_progress_partial() {
        let mut progress = UserProgress::new();
        progress.reset_daily_goals();

        // Complete one goal
        for _ in 0..10 {
            progress.daily_goals[0].increment();
        }

        let pct = progress.daily_goal_progress();
        assert!(pct > 30.0 && pct < 40.0); // ~33%
    }

    #[test]
    fn test_daily_goal_progress_all_complete() {
        let mut progress = UserProgress::new();
        progress.reset_daily_goals();

        // Complete all goals
        for goal in &mut progress.daily_goals {
            for _ in 0..goal.target {
                goal.increment();
            }
        }

        assert_eq!(progress.daily_goal_progress(), 100.0);
    }

    // ========== level_progress tests ==========

    #[test]
    fn test_level_progress_method() {
        let mut progress = UserProgress::new();
        progress.award_xp(150);

        let lp = progress.level_progress();
        assert_eq!(lp.level, 2);
        assert_eq!(lp.current_xp, 150);
    }

    // ========== achievement_count tests ==========

    #[test]
    fn test_achievement_count() {
        let mut progress = UserProgress::new();
        assert_eq!(progress.achievement_count(), 0);

        progress.unlock_achievement("test1", AchievementTier::Bronze);
        assert_eq!(progress.achievement_count(), 1);

        progress.unlock_achievement("test2", AchievementTier::Silver);
        assert_eq!(progress.achievement_count(), 2);
    }

    // ========== UserStats additional tests ==========

    #[test]
    fn test_user_stats_record_algorithm() {
        let mut stats = UserStats::default();
        stats.record_algorithm();
        assert_eq!(stats.algorithms_run, 1);

        stats.record_algorithm();
        assert_eq!(stats.algorithms_run, 2);
    }

    #[test]
    fn test_user_stats_record_vector_search() {
        let mut stats = UserStats::default();
        stats.record_vector_search();
        assert_eq!(stats.vector_searches, 1);
    }

    #[test]
    fn test_user_stats_record_graph_traversal() {
        let mut stats = UserStats::default();
        stats.record_graph_traversal();
        assert_eq!(stats.graph_traversals, 1);
    }

    #[test]
    fn test_user_stats_record_path_found() {
        let mut stats = UserStats::default();
        stats.record_path_found();
        assert_eq!(stats.paths_found, 1);
    }

    #[test]
    fn test_user_stats_record_community_detected() {
        let mut stats = UserStats::default();
        stats.record_community_detected();
        assert_eq!(stats.communities_detected, 1);
    }

    #[test]
    fn test_user_stats_serialization() {
        let mut stats = UserStats::default();
        stats.record_query(5.0);
        stats.record_algorithm();

        let json = serde_json::to_string(&stats).expect("serialization failed");
        assert!(json.contains("queries_total"));
        assert!(json.contains("algorithms_run"));
    }

    // ========== DailyGoal additional tests ==========

    #[test]
    fn test_daily_goal_progress_overflow() {
        let mut goal = DailyGoal::new(GoalType::Queries, 5);
        for _ in 0..10 {
            goal.increment();
        }
        // Progress should cap at 100%
        assert_eq!(goal.progress_percent(), 100.0);
    }

    #[test]
    fn test_goal_type_fast_queries() {
        assert_eq!(GoalType::FastQueries.display_name(), "Fast Queries");
    }

    #[test]
    fn test_goal_type_serialization() {
        let goal_type = GoalType::Algorithms;
        let json = serde_json::to_string(&goal_type).expect("serialization failed");
        assert!(json.contains("algorithms"));
    }

    #[test]
    fn test_daily_goal_serialization() {
        let goal = DailyGoal::new(GoalType::Queries, 10);
        let json = serde_json::to_string(&goal).expect("serialization failed");
        assert!(json.contains("goal_type"));
        assert!(json.contains("target"));
    }

    // ========== Additional GoalType tests ==========

    #[test]
    fn test_goal_type_vector_searches() {
        assert_eq!(GoalType::VectorSearches.display_name(), "Vector Searches");
    }

    #[test]
    fn test_goal_type_paths_found() {
        assert_eq!(GoalType::PathsFound.display_name(), "Find Paths");
    }

    #[test]
    fn test_goal_type_all_serialization() {
        for goal_type in [
            GoalType::Queries,
            GoalType::Algorithms,
            GoalType::FastQueries,
            GoalType::VectorSearches,
            GoalType::PathsFound,
        ] {
            let json = serde_json::to_string(&goal_type).expect("serialization failed");
            let decoded: GoalType = serde_json::from_str(&json).expect("deserialization failed");
            assert_eq!(decoded, goal_type);
        }
    }

    #[test]
    fn test_goal_type_debug() {
        let goal_type = GoalType::Algorithms;
        let debug_str = format!("{:?}", goal_type);
        assert!(debug_str.contains("Algorithms"));
    }

    // ========== Additional DailyGoal tests ==========

    #[test]
    fn test_daily_goal_zero_target() {
        let goal = DailyGoal::new(GoalType::Queries, 0);
        assert!(goal.is_complete());
        assert_eq!(goal.progress_percent(), 100.0);
    }

    #[test]
    fn test_daily_goal_increment_saturating() {
        let mut goal = DailyGoal::new(GoalType::Queries, u32::MAX);
        goal.current = u32::MAX - 1;
        goal.increment();
        assert_eq!(goal.current, u32::MAX);
        goal.increment(); // Should not overflow
        assert_eq!(goal.current, u32::MAX);
    }

    #[test]
    fn test_daily_goal_clone() {
        let goal = DailyGoal::new(GoalType::FastQueries, 20);
        let cloned = goal.clone();
        assert_eq!(cloned.goal_type, GoalType::FastQueries);
        assert_eq!(cloned.target, 20);
    }

    #[test]
    fn test_daily_goal_debug() {
        let goal = DailyGoal::new(GoalType::Queries, 10);
        let debug_str = format!("{:?}", goal);
        assert!(debug_str.contains("DailyGoal"));
        assert!(debug_str.contains("Queries"));
    }

    #[test]
    fn test_daily_goal_deserialization() {
        let goal = DailyGoal::new(GoalType::Algorithms, 5);
        let json = serde_json::to_string(&goal).expect("serialization failed");
        let decoded: DailyGoal = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.goal_type, GoalType::Algorithms);
        assert_eq!(decoded.target, 5);
    }

    // ========== Additional UserStats tests ==========

    #[test]
    fn test_user_stats_record_query_slow() {
        let mut stats = UserStats::default();
        stats.record_query(100.0); // Slow query
        assert_eq!(stats.queries_total, 1);
        assert_eq!(stats.queries_fast, 0);
        assert_eq!(stats.queries_lightning, 0);
    }

    #[test]
    fn test_user_stats_record_query_boundary() {
        let mut stats = UserStats::default();
        stats.record_query(10.0); // Exactly at boundary
        assert_eq!(stats.queries_fast, 0); // < 10.0, not <= 10.0

        stats.record_query(1.0); // Exactly at lightning boundary
        assert_eq!(stats.queries_lightning, 0); // < 1.0, not <= 1.0
    }

    #[test]
    fn test_user_stats_clone() {
        let mut stats = UserStats::default();
        stats.record_query(0.5);
        stats.record_algorithm();
        let cloned = stats.clone();
        assert_eq!(cloned.queries_total, 1);
        assert_eq!(cloned.algorithms_run, 1);
    }

    #[test]
    fn test_user_stats_debug() {
        let stats = UserStats::default();
        let debug_str = format!("{:?}", stats);
        assert!(debug_str.contains("UserStats"));
        assert!(debug_str.contains("queries_total"));
    }

    #[test]
    fn test_user_stats_deserialization() {
        let mut stats = UserStats::default();
        stats.record_query(5.0);
        let json = serde_json::to_string(&stats).expect("serialization failed");
        let decoded: UserStats = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.queries_total, 1);
    }

    // ========== Additional UserProgress tests ==========

    #[test]
    fn test_user_progress_award_xp_overflow() {
        let mut progress = UserProgress::new();
        progress.xp_total = u64::MAX - 10;
        progress.award_xp(20); // Should saturate
        assert_eq!(progress.xp_total, u64::MAX);
    }

    #[test]
    fn test_user_progress_clone() {
        let mut progress = UserProgress::new();
        progress.award_xp(100);
        progress.unlock_achievement("test", AchievementTier::Bronze);
        let cloned = progress.clone();
        assert_eq!(cloned.xp_total, progress.xp_total);
        assert!(cloned.has_achievement("test"));
    }

    #[test]
    fn test_user_progress_debug() {
        let progress = UserProgress::new();
        let debug_str = format!("{:?}", progress);
        assert!(debug_str.contains("UserProgress"));
        assert!(debug_str.contains("xp_total"));
    }

    #[test]
    fn test_user_progress_deserialization() {
        let mut progress = UserProgress::new();
        progress.award_xp(500);
        let json = serde_json::to_string(&progress).expect("serialization failed");
        let decoded: UserProgress = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.xp_total, 500);
    }

    #[test]
    fn test_is_consecutive_month_not_first() {
        let mut progress = UserProgress::new();
        progress.last_activity_date = 20240130; // Jan 30
                                                // Not first of month check
        assert!(!progress.is_consecutive_month(20240202)); // Feb 2
    }

    #[test]
    fn test_is_consecutive_month_different_year_non_december() {
        let mut progress = UserProgress::new();
        progress.last_activity_date = 20231130; // Nov 30, 2023
                                                // Different year but not Dec->Jan
        assert!(!progress.is_consecutive_month(20240101)); // Jan 1, 2024
    }
}
