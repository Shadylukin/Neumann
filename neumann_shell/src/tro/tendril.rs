// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tendril system for organic border growth patterns.
//!
//! Tendrils extend from the border into the content area, creating
//! branching organic patterns that react to database activity.

// Visual calculations involve many float/int conversions with acceptable precision loss.
#![allow(clippy::cast_precision_loss)]

use super::border_region::BorderCoord;

/// A single tendril extending from the border.
#[derive(Debug, Clone)]
pub struct Tendril {
    /// Origin point on the border.
    pub origin: BorderCoord,
    /// Current tip position.
    pub tip: BorderCoord,
    /// Trail of positions with intensity (position, intensity).
    pub trail: Vec<(BorderCoord, f32)>,
    /// Age in frames.
    pub age: u32,
    /// Growth direction in radians.
    pub direction: f32,
    /// Current intensity (fades over time).
    pub intensity: f32,
    /// Whether the tendril is still growing.
    pub growing: bool,
}

impl Tendril {
    /// Creates a new tendril at the given origin.
    #[must_use]
    pub fn new(origin: BorderCoord, direction: f32, intensity: f32) -> Self {
        Self {
            origin,
            tip: origin,
            trail: vec![(origin, intensity)],
            age: 0,
            direction,
            intensity,
            growing: true,
        }
    }

    /// Returns the length of the tendril trail.
    #[must_use]
    pub fn length(&self) -> usize {
        self.trail.len()
    }

    /// Checks if the tendril is effectively dead (very low intensity).
    #[must_use]
    pub fn is_dead(&self) -> bool {
        self.intensity < 0.01 && !self.growing
    }
}

/// Configuration for tendril growth behavior.
#[derive(Debug, Clone)]
pub struct TendrilConfig {
    /// Maximum length in cells from border.
    pub max_length: usize,
    /// Pheromone level threshold to spawn new tendril.
    pub spawn_threshold: f32,
    /// Growth speed in cells per frame.
    pub growth_speed: f32,
    /// Intensity decay rate per frame (0.0-1.0).
    pub decay_rate: f32,
    /// Trail fade rate per frame (0.0-1.0).
    pub trail_fade: f32,
    /// Maximum concurrent tendrils.
    pub max_tendrils: usize,
    /// Heat attraction strength (steering towards hot areas).
    pub heat_attraction: f32,
    /// Randomness in direction (radians).
    pub direction_noise: f32,
    /// Minimum frames between spawns at same location.
    pub spawn_cooldown: u32,
}

impl Default for TendrilConfig {
    fn default() -> Self {
        Self {
            max_length: 8,
            spawn_threshold: 0.7,
            growth_speed: 0.3,
            decay_rate: 0.98,
            trail_fade: 0.95,
            max_tendrils: 20,
            heat_attraction: 0.5,
            direction_noise: 0.3,
            spawn_cooldown: 30,
        }
    }
}

impl TendrilConfig {
    /// Creates a minimal config for low-resource environments.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            max_length: 4,
            spawn_threshold: 0.8,
            growth_speed: 0.2,
            decay_rate: 0.95,
            trail_fade: 0.9,
            max_tendrils: 10,
            heat_attraction: 0.3,
            direction_noise: 0.2,
            spawn_cooldown: 60,
        }
    }

    /// Creates an aggressive config for dramatic effects.
    #[must_use]
    pub fn aggressive() -> Self {
        Self {
            max_length: 12,
            spawn_threshold: 0.5,
            growth_speed: 0.5,
            decay_rate: 0.99,
            trail_fade: 0.97,
            max_tendrils: 40,
            heat_attraction: 0.8,
            direction_noise: 0.4,
            spawn_cooldown: 15,
        }
    }
}

/// Manages tendril lifecycle and rendering.
pub struct TendrilManager {
    config: TendrilConfig,
    tendrils: Vec<Tendril>,
    spawn_cooldowns: Vec<(BorderCoord, u32)>,
    frame: u32,
    terminal_width: u16,
    terminal_height: u16,
}

impl TendrilManager {
    /// Creates a new tendril manager.
    #[must_use]
    pub fn new(config: TendrilConfig, width: u16, height: u16) -> Self {
        let capacity = config.max_tendrils;
        Self {
            config,
            tendrils: Vec::with_capacity(capacity),
            spawn_cooldowns: Vec::new(),
            frame: 0,
            terminal_width: width,
            terminal_height: height,
        }
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &TendrilConfig {
        &self.config
    }

    /// Updates the configuration.
    pub fn set_config(&mut self, config: TendrilConfig) {
        self.config = config;
    }

    /// Resizes the manager for new terminal dimensions.
    pub fn resize(&mut self, width: u16, height: u16) {
        self.terminal_width = width;
        self.terminal_height = height;

        // Prune tendrils that are now out of bounds
        self.tendrils.retain(|t| {
            t.tip.x < width && t.tip.y < height && t.origin.x < width && t.origin.y < height
        });
    }

    /// Updates all tendrils (grow, decay, prune).
    pub fn update(&mut self, pheromone: &[f32], heat: &[f32]) {
        self.frame = self.frame.wrapping_add(1);

        // Update cooldowns
        self.spawn_cooldowns.retain_mut(|(_, cooldown)| {
            *cooldown = cooldown.saturating_sub(1);
            *cooldown > 0
        });

        // Try to spawn new tendrils from high-pheromone areas
        self.spawn_new(pheromone);

        // Grow existing tendrils
        self.grow(heat);

        // Decay all tendrils
        self.decay();

        // Remove dead tendrils
        self.prune();
    }

    /// Attempts to spawn new tendrils from border cells with high pheromone.
    fn spawn_new(&mut self, pheromone: &[f32]) {
        if self.tendrils.len() >= self.config.max_tendrils {
            return;
        }

        // Find high-pheromone border positions
        for (idx, &level) in pheromone.iter().enumerate() {
            if level < self.config.spawn_threshold {
                continue;
            }

            if self.tendrils.len() >= self.config.max_tendrils {
                break;
            }

            // Convert linear index to coordinate
            if let Some(coord) = self.linear_to_coord(idx) {
                // Check cooldown
                if self.spawn_cooldowns.iter().any(|(c, _)| *c == coord) {
                    continue;
                }

                // Determine direction (towards center)
                let direction = self.direction_towards_center(coord);

                // Spawn tendril
                let tendril = Tendril::new(coord, direction, level);
                self.tendrils.push(tendril);

                // Add cooldown
                self.spawn_cooldowns
                    .push((coord, self.config.spawn_cooldown));
            }
        }
    }

    /// Grows active tendrils.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn grow(&mut self, heat: &[f32]) {
        // Extract values needed for heat steering to avoid borrow conflicts
        let terminal_width = self.terminal_width;
        let terminal_height = self.terminal_height;
        let heat_attraction = self.config.heat_attraction;
        let max_length = self.config.max_length;
        let growth_speed = self.config.growth_speed;
        let direction_noise = self.config.direction_noise;

        for tendril in &mut self.tendrils {
            if !tendril.growing {
                continue;
            }

            // Check if max length reached
            if tendril.length() >= max_length {
                tendril.growing = false;
                continue;
            }

            // Accumulate growth
            tendril.age += 1;
            let growth_progress = tendril.age as f32 * growth_speed;

            if growth_progress >= tendril.length() as f32 {
                // Calculate new tip position
                let noise = (rand::random::<f32>() - 0.5) * direction_noise;
                let mut new_direction = tendril.direction + noise;

                // Apply heat attraction if available
                if !heat.is_empty() {
                    new_direction += Self::heat_steering_static(
                        tendril.tip,
                        heat,
                        terminal_width,
                        terminal_height,
                        heat_attraction,
                    );
                }

                tendril.direction = new_direction;

                let dx = new_direction.cos();
                let dy = new_direction.sin();

                let new_x = (f32::from(tendril.tip.x) + dx).round();
                let new_y = (f32::from(tendril.tip.y) + dy).round();

                // Check bounds
                if new_x >= 0.0
                    && new_x < f32::from(terminal_width)
                    && new_y >= 0.0
                    && new_y < f32::from(terminal_height)
                {
                    let new_tip = BorderCoord::new(new_x as u16, new_y as u16);

                    // Don't grow back to same position
                    if new_tip != tendril.tip {
                        tendril.tip = new_tip;
                        tendril.trail.push((new_tip, tendril.intensity));
                    }
                } else {
                    // Hit boundary, stop growing
                    tendril.growing = false;
                }
            }
        }
    }

    /// Decays tendril intensities.
    fn decay(&mut self) {
        for tendril in &mut self.tendrils {
            tendril.intensity *= self.config.decay_rate;

            // Fade trail
            for (_, intensity) in &mut tendril.trail {
                *intensity *= self.config.trail_fade;
            }

            // Stop growing if too faint
            if tendril.intensity < 0.1 {
                tendril.growing = false;
            }
        }
    }

    /// Removes dead tendrils.
    fn prune(&mut self) {
        self.tendrils.retain(|t| !t.is_dead());
    }

    /// Returns all visible tendril cells for rendering.
    pub fn visible_cells(&self) -> impl Iterator<Item = (BorderCoord, f32)> + '_ {
        self.tendrils.iter().flat_map(|t| t.trail.iter().copied())
    }

    /// Returns the number of active tendrils.
    #[must_use]
    pub fn count(&self) -> usize {
        self.tendrils.len()
    }

    /// Clears all tendrils.
    pub fn clear(&mut self) {
        self.tendrils.clear();
        self.spawn_cooldowns.clear();
    }

    /// Calculates direction from border coordinate towards terminal center.
    fn direction_towards_center(&self, coord: BorderCoord) -> f32 {
        let center_x = f32::from(self.terminal_width) / 2.0;
        let center_y = f32::from(self.terminal_height) / 2.0;

        let dx = center_x - f32::from(coord.x);
        let dy = center_y - f32::from(coord.y);

        dy.atan2(dx)
    }

    /// Calculates steering adjustment based on nearby heat (static version).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn heat_steering_static(
        tip: BorderCoord,
        heat: &[f32],
        terminal_width: u16,
        terminal_height: u16,
        heat_attraction: f32,
    ) -> f32 {
        let perimeter = Self::calculate_perimeter_static(terminal_width, terminal_height);
        if perimeter == 0 || heat.is_empty() {
            return 0.0;
        }

        // Sample heat in a small area around the tip
        let mut total_pull = 0.0;
        let sample_radius = 3i32;

        for dy in -sample_radius..=sample_radius {
            for dx in -sample_radius..=sample_radius {
                if dx == 0 && dy == 0 {
                    continue;
                }

                let sample_x = i32::from(tip.x) + dx;
                let sample_y = i32::from(tip.y) + dy;

                if sample_x >= 0
                    && sample_x < i32::from(terminal_width)
                    && sample_y >= 0
                    && sample_y < i32::from(terminal_height)
                {
                    let sample_coord = BorderCoord::new(sample_x as u16, sample_y as u16);
                    if let Some(idx) =
                        Self::coord_to_linear_static(sample_coord, terminal_width, terminal_height)
                    {
                        if let Some(&h) = heat.get(idx) {
                            let angle = (dy as f32).atan2(dx as f32);
                            total_pull += angle * h * heat_attraction;
                        }
                    }
                }
            }
        }

        total_pull.clamp(-0.5, 0.5)
    }

    /// Converts linear border index to coordinate.
    #[allow(clippy::cast_possible_truncation)]
    fn linear_to_coord(&self, index: usize) -> Option<BorderCoord> {
        let w = self.terminal_width as usize;
        let h = self.terminal_height as usize;

        if w < 2 || h < 2 {
            return None;
        }

        let top_len = w;
        let right_len = h.saturating_sub(2);
        let bottom_len = w;
        let total = top_len + right_len + bottom_len + h.saturating_sub(2);

        if index >= total {
            return None;
        }

        if index < top_len {
            Some(BorderCoord::new(index as u16, 0))
        } else if index < top_len + right_len {
            let offset = index - top_len;
            Some(BorderCoord::new(
                self.terminal_width - 1,
                (offset + 1) as u16,
            ))
        } else if index < top_len + right_len + bottom_len {
            let offset = index - top_len - right_len;
            Some(BorderCoord::new(
                (w - 1 - offset) as u16,
                self.terminal_height - 1,
            ))
        } else {
            let offset = index - top_len - right_len - bottom_len;
            Some(BorderCoord::new(0, (h - 2 - offset) as u16))
        }
    }

    /// Converts coordinate to linear border index.
    fn coord_to_linear(&self, coord: BorderCoord) -> Option<usize> {
        let w = self.terminal_width;
        let h = self.terminal_height;
        let x = coord.x;
        let y = coord.y;

        if w < 2 || h < 2 {
            return None;
        }

        if y == 0 && x < w {
            Some(x as usize)
        } else if x == w - 1 && y > 0 && y < h - 1 {
            Some(w as usize + (y - 1) as usize)
        } else if y == h - 1 && x < w {
            Some(w as usize + (h - 2) as usize + (w - 1 - x) as usize)
        } else if x == 0 && y > 0 && y < h - 1 {
            Some(w as usize + (h - 2) as usize + w as usize + (h - 2 - y) as usize)
        } else {
            None
        }
    }

    /// Calculates the border perimeter.
    fn calculate_perimeter(&self) -> usize {
        Self::calculate_perimeter_static(self.terminal_width, self.terminal_height)
    }

    /// Calculates the border perimeter (static version).
    fn calculate_perimeter_static(terminal_width: u16, terminal_height: u16) -> usize {
        let w = terminal_width as usize;
        let h = terminal_height as usize;
        if w < 2 || h < 2 {
            0
        } else {
            2 * w + 2 * h.saturating_sub(2)
        }
    }

    /// Converts coordinate to linear border index (static version).
    fn coord_to_linear_static(
        coord: BorderCoord,
        terminal_width: u16,
        terminal_height: u16,
    ) -> Option<usize> {
        let w = terminal_width;
        let h = terminal_height;
        let x = coord.x;
        let y = coord.y;

        if w < 2 || h < 2 {
            return None;
        }

        if y == 0 && x < w {
            Some(x as usize)
        } else if x == w - 1 && y > 0 && y < h - 1 {
            Some(w as usize + (y - 1) as usize)
        } else if y == h - 1 && x < w {
            Some(w as usize + (h - 2) as usize + (w - 1 - x) as usize)
        } else if x == 0 && y > 0 && y < h - 1 {
            Some(w as usize + (h - 2) as usize + w as usize + (h - 2 - y) as usize)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tendril_new() {
        let origin = BorderCoord::new(10, 0);
        let tendril = Tendril::new(origin, 1.57, 0.8);

        assert_eq!(tendril.origin, origin);
        assert_eq!(tendril.tip, origin);
        assert_eq!(tendril.length(), 1);
        assert!(tendril.growing);
    }

    #[test]
    fn test_tendril_is_dead() {
        let mut tendril = Tendril::new(BorderCoord::new(0, 0), 0.0, 0.5);
        assert!(!tendril.is_dead());

        tendril.intensity = 0.005;
        tendril.growing = false;
        assert!(tendril.is_dead());
    }

    #[test]
    fn test_tendril_config_default() {
        let config = TendrilConfig::default();
        assert_eq!(config.max_length, 8);
        assert_eq!(config.max_tendrils, 20);
    }

    #[test]
    fn test_tendril_config_minimal() {
        let config = TendrilConfig::minimal();
        assert_eq!(config.max_length, 4);
        assert_eq!(config.max_tendrils, 10);
    }

    #[test]
    fn test_tendril_config_aggressive() {
        let config = TendrilConfig::aggressive();
        assert_eq!(config.max_length, 12);
        assert_eq!(config.max_tendrils, 40);
    }

    #[test]
    fn test_tendril_manager_new() {
        let manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_tendril_manager_resize() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        manager.resize(100, 30);
        assert_eq!(manager.terminal_width, 100);
        assert_eq!(manager.terminal_height, 30);
    }

    #[test]
    fn test_tendril_manager_clear() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        // Manually add a tendril
        manager
            .tendrils
            .push(Tendril::new(BorderCoord::new(0, 0), 0.0, 1.0));
        assert_eq!(manager.count(), 1);

        manager.clear();
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_tendril_manager_visible_cells() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        manager
            .tendrils
            .push(Tendril::new(BorderCoord::new(10, 0), 0.0, 0.8));

        let cells: Vec<_> = manager.visible_cells().collect();
        assert!(!cells.is_empty());
    }

    #[test]
    fn test_tendril_manager_direction_towards_center() {
        let manager = TendrilManager::new(TendrilConfig::default(), 80, 24);

        // Top-left corner should point towards center (down-right)
        let dir = manager.direction_towards_center(BorderCoord::new(0, 0));
        assert!(dir > 0.0); // Positive angle (towards bottom-right)

        // Bottom-right corner should point towards center (up-left)
        let dir2 = manager.direction_towards_center(BorderCoord::new(79, 23));
        assert!(dir2 < 0.0); // Negative angle (towards top-left)
    }

    #[test]
    fn test_tendril_manager_linear_to_coord() {
        let manager = TendrilManager::new(TendrilConfig::default(), 80, 24);

        assert_eq!(manager.linear_to_coord(0), Some(BorderCoord::new(0, 0)));
        assert_eq!(manager.linear_to_coord(79), Some(BorderCoord::new(79, 0)));
        assert_eq!(manager.linear_to_coord(80), Some(BorderCoord::new(79, 1)));
    }

    #[test]
    fn test_tendril_manager_update_empty() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        manager.update(&[], &[]);
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_tendril_manager_update_spawns() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);

        // High pheromone at position 10
        let mut pheromone = vec![0.0; 204]; // 80*2 + 22*2 = 204
        pheromone[10] = 0.9;

        manager.update(&pheromone, &[]);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_tendril_manager_spawn_cooldown() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);

        let mut pheromone = vec![0.0; 204];
        pheromone[10] = 0.9;

        manager.update(&pheromone, &[]);
        assert_eq!(manager.count(), 1);

        // Same position should not spawn again due to cooldown
        manager.update(&pheromone, &[]);
        assert_eq!(manager.count(), 1);
    }

    #[test]
    fn test_tendril_manager_max_tendrils() {
        let config = TendrilConfig {
            max_tendrils: 3,
            spawn_threshold: 0.5,
            ..TendrilConfig::default()
        };
        let mut manager = TendrilManager::new(config, 80, 24);

        // High pheromone everywhere
        let pheromone = vec![0.9; 204];

        manager.update(&pheromone, &[]);
        assert!(manager.count() <= 3);
    }

    #[test]
    fn test_tendril_decay() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);
        manager
            .tendrils
            .push(Tendril::new(BorderCoord::new(0, 0), 0.0, 1.0));

        let initial = manager.tendrils[0].intensity;
        manager.decay();
        assert!(manager.tendrils[0].intensity < initial);
    }

    #[test]
    fn test_tendril_prune() {
        let mut manager = TendrilManager::new(TendrilConfig::default(), 80, 24);

        // Add a dead tendril
        let mut tendril = Tendril::new(BorderCoord::new(0, 0), 0.0, 0.005);
        tendril.growing = false;
        manager.tendrils.push(tendril);

        manager.prune();
        assert_eq!(manager.count(), 0);
    }
}
