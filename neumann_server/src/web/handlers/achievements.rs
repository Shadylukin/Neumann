// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Achievements page handlers for the admin UI.
//!
//! Displays user progress, achievements, XP, levels, and daily goals.

use std::sync::Arc;

use axum::extract::State;
use maud::{html, Markup, PreEscaped};

use crate::gamification::{
    level_title, Achievement, AchievementCategory, AchievementTier, DailyGoal, UserProgress,
    ACHIEVEMENTS,
};
use crate::web::templates::{layout, page_header};
use crate::web::{AdminContext, NavItem};

/// Achievements dashboard page.
pub async fn dashboard(State(_ctx): State<Arc<AdminContext>>) -> Markup {
    // In a real implementation, this would load from persistent storage
    let mut progress = UserProgress::new();
    progress.award_xp(1250);
    progress.unlock_achievement("first_query", AchievementTier::Bronze);
    progress.unlock_achievement("first_graph", AchievementTier::Bronze);
    progress.unlock_achievement("fast_query", AchievementTier::Bronze);
    progress.streak_current = 5;
    progress.streak_best = 12;
    progress.reset_daily_goals();
    progress.daily_goals[0].current = 7;
    progress.daily_goals[1].current = 2;

    let level_info = progress.level_progress();

    let content = html! {
        (page_header("ACHIEVEMENTS", Some("Track your progress and unlock rewards")))

        // Level and XP bar
        div class="terminal-panel mb-6" {
            div class="panel-header" { "OPERATOR STATUS" }
            div class="panel-content" {
                div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4" {
                    // Level info
                    div class="flex items-center gap-4" {
                        div class="text-4xl font-display text-phosphor glow-phosphor" {
                            "L" (level_info.level)
                        }
                        div {
                            div class="font-terminal text-phosphor" {
                                (level_title(level_info.level))
                            }
                            div class="text-sm text-phosphor-dim font-terminal" {
                                (format!("{} / {} XP", level_info.current_xp,
                                    if level_info.is_max_level { "MAX".to_string() }
                                    else { (level_info.current_xp + level_info.xp_for_level - level_info.xp_in_level).to_string() }
                                ))
                            }
                        }
                    }
                    // XP bar
                    div class="flex-1 max-w-md" {
                        (xp_bar(&level_info))
                    }
                }
            }
        }

        // Stats row
        div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6" {
            (stat_card("LEVEL", &level_info.level.to_string(), "Current rank"))
            (stat_card("XP", &format_number(level_info.current_xp), "Total earned"))
            (stat_card("STREAK", &format!("{} days", progress.streak_current),
                &format!("Best: {} days", progress.streak_best)))
            (stat_card("ACHIEVEMENTS", &format!("{}/{}", progress.achievement_count(), ACHIEVEMENTS.len()),
                "Unlocked"))
        }

        // Daily goals
        div class="terminal-panel mb-6" {
            div class="panel-header" { "DAILY OBJECTIVES" }
            div class="panel-content" {
                div class="grid grid-cols-1 md:grid-cols-3 gap-4" {
                    @for goal in &progress.daily_goals {
                        (daily_goal_card(goal))
                    }
                }
            }
            div class="panel-footer" {
                span class="text-phosphor-dim" {
                    "Complete all objectives for bonus XP"
                }
            }
        }

        // Achievement categories
        div class="space-y-6" {
            @for category in [AchievementCategory::Discovery, AchievementCategory::Performance,
                              AchievementCategory::Dedication, AchievementCategory::Mastery] {
                (achievement_category_section(category, &progress))
            }
        }

        // Achievement celebration script
        script { (PreEscaped(r#"
            // Achievement unlock animation
            function celebrateAchievement(name) {
                const toast = document.createElement('div');
                toast.className = 'achievement-toast';
                toast.innerHTML = `
                    <div class="achievement-toast-content">
                        <span class="achievement-icon">*</span>
                        <span>Achievement Unlocked: ${name}</span>
                    </div>
                `;
                document.body.appendChild(toast);
                setTimeout(() => toast.remove(), 3000);
            }
        "#)) }
    };

    layout("Achievements", NavItem::Dashboard, content)
}

/// Renders the XP progress bar.
fn xp_bar(level_info: &crate::gamification::LevelProgress) -> Markup {
    html! {
        div {
            div class="flex justify-between mb-1 text-xs font-terminal" {
                span class="text-phosphor-dim" { "XP Progress" }
                span class="text-phosphor" {
                    @if level_info.is_max_level {
                        "MAX LEVEL"
                    } @else {
                        (format!("{:.0}%", level_info.percentage))
                    }
                }
            }
            div class="h-3 bg-soot-gray border border-phosphor-dark relative overflow-hidden" {
                div class="h-full bg-gradient-to-r from-phosphor-dark via-phosphor-dim to-phosphor transition-all duration-500"
                    style=(format!("width: {}%", level_info.percentage)) {}
                // Glow effect
                div class="absolute inset-0 bg-gradient-to-r from-transparent via-phosphor to-transparent opacity-30"
                    style="animation: xp-shimmer 2s infinite linear;" {}
            }
        }
        style { (PreEscaped(r"
            @keyframes xp-shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
        ")) }
    }
}

/// Renders a stat card.
fn stat_card(label: &str, value: &str, subtitle: &str) -> Markup {
    html! {
        div class="terminal-panel" {
            div class="panel-content text-center" {
                div class="text-2xl font-data text-phosphor glow-phosphor" {
                    (value)
                }
                div class="text-sm font-terminal text-phosphor-dim mt-1" {
                    (label)
                }
                div class="text-xs font-terminal text-phosphor-dark mt-1" {
                    (subtitle)
                }
            }
        }
    }
}

/// Renders a daily goal card.
fn daily_goal_card(goal: &DailyGoal) -> Markup {
    let complete = goal.is_complete();
    let progress = goal.progress_percent();

    html! {
        div class=(format!("terminal-panel {}", if complete { "border-phosphor" } else { "" })) {
            div class="panel-content" {
                div class="flex justify-between items-center mb-2" {
                    span class="font-terminal text-sm" {
                        (goal.goal_type.display_name())
                    }
                    @if complete {
                        span class="text-phosphor glow-phosphor" { "[COMPLETE]" }
                    } @else {
                        span class="text-phosphor-dim" {
                            (format!("{}/{}", goal.current, goal.target))
                        }
                    }
                }
                div class="h-2 bg-soot-gray border border-phosphor-dark" {
                    div class=(format!("h-full transition-all duration-300 {}",
                        if complete { "bg-phosphor" } else { "bg-phosphor-dim" }))
                        style=(format!("width: {}%", progress)) {}
                }
            }
        }
    }
}

/// Renders an achievement category section.
fn achievement_category_section(category: AchievementCategory, progress: &UserProgress) -> Markup {
    let achievements: Vec<_> = ACHIEVEMENTS
        .iter()
        .filter(|a| a.category == category && !a.hidden)
        .collect();

    if achievements.is_empty() {
        return html! {};
    }

    html! {
        div class="terminal-panel" {
            div class="panel-header" {
                span { (category.icon()) " " (category.display_name().to_uppercase()) }
            }
            div class="panel-content" {
                div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3" {
                    @for achievement in achievements {
                        (achievement_card(achievement, progress.has_achievement(achievement.id)))
                    }
                }
            }
        }
    }
}

/// Renders an achievement card.
fn achievement_card(achievement: &Achievement, unlocked: bool) -> Markup {
    let tier_class = achievement.tier.css_class();
    let opacity = if unlocked { "" } else { "opacity-50" };

    html! {
        div class=(format!("border p-3 {} {}", tier_class, opacity)) {
            div class="flex justify-between items-start mb-2" {
                div class="font-terminal text-sm" {
                    @if unlocked {
                        span class="text-phosphor" { "[*] " }
                    } @else {
                        span class="text-phosphor-dark" { "[ ] " }
                    }
                    (achievement.name)
                }
                span class=(format!("text-xs font-terminal {}", tier_color(achievement.tier))) {
                    (achievement.tier.display_name())
                }
            }
            div class="text-xs text-phosphor-dim font-terminal" {
                (achievement.description)
            }
            @if let Some(threshold) = achievement.threshold {
                div class="text-xs text-phosphor-dark font-terminal mt-1" {
                    "Target: " (threshold)
                }
            }
        }
    }
}

/// Returns the color class for a tier.
fn tier_color(tier: AchievementTier) -> &'static str {
    match tier {
        AchievementTier::Bronze => "text-amber-dim",
        AchievementTier::Silver => "text-phosphor-dim",
        AchievementTier::Gold => "text-amber",
        AchievementTier::Platinum => "text-phosphor",
    }
}

/// Formats a number with thousand separators.
fn format_number(n: u64) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamification::{level_progress, GoalType};

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_tier_color() {
        assert_eq!(tier_color(AchievementTier::Bronze), "text-amber-dim");
        assert_eq!(tier_color(AchievementTier::Platinum), "text-phosphor");
    }

    #[test]
    fn test_xp_bar_rendering() {
        let level_info = level_progress(150);
        let html = xp_bar(&level_info).into_string();
        assert!(html.contains("XP Progress"));
        assert!(html.contains('%'));
    }

    #[test]
    fn test_stat_card_rendering() {
        let html = stat_card("TEST", "42", "subtitle").into_string();
        assert!(html.contains("TEST"));
        assert!(html.contains("42"));
        assert!(html.contains("subtitle"));
    }

    #[test]
    fn test_daily_goal_card_incomplete() {
        let goal = DailyGoal::new(GoalType::Queries, 10);
        let html = daily_goal_card(&goal).into_string();
        assert!(html.contains("Execute Queries"));
        assert!(html.contains("0/10"));
        assert!(!html.contains("COMPLETE"));
    }

    #[test]
    fn test_daily_goal_card_complete() {
        let mut goal = DailyGoal::new(GoalType::Queries, 10);
        goal.current = 10;
        let html = daily_goal_card(&goal).into_string();
        assert!(html.contains("COMPLETE"));
    }

    #[test]
    fn test_achievement_card_unlocked() {
        let achievement = &ACHIEVEMENTS[0];
        let html = achievement_card(achievement, true).into_string();
        assert!(html.contains("[*]"));
        assert!(html.contains(achievement.name));
    }

    #[test]
    fn test_achievement_card_locked() {
        let achievement = &ACHIEVEMENTS[0];
        let html = achievement_card(achievement, false).into_string();
        assert!(html.contains("[ ]"));
        assert!(html.contains("opacity-50"));
    }

    #[test]
    fn test_achievement_category_section() {
        let progress = UserProgress::new();
        let html =
            achievement_category_section(AchievementCategory::Discovery, &progress).into_string();
        assert!(html.contains("DISCOVERY"));
    }
}
