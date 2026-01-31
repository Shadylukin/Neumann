// SPDX-License-Identifier: MIT OR Apache-2.0
//! TRO organism state machine and cell definitions.

// This module performs many float/int conversions for visual calculations.
// Precision loss is acceptable for small array indices (< 100 elements).
// u8 to f32 casts are also intentional for color manipulation.
#![allow(
    clippy::cast_precision_loss,
    clippy::suboptimal_flops,
    clippy::cast_lossless
)]

use super::archetype::ArchetypeRegistry;
use super::border_region::{BorderRegion, ResolutionMode};
use super::charset::CharsetMode;
use super::palette::Palette;
use super::physarum::PhysarumField;
use super::reaction::ReactionField;
use super::tendril::{TendrilConfig, TendrilManager};
use smallvec::SmallVec;

/// A single cell in the TRO border.
#[derive(Debug, Clone)]
pub struct TroCell {
    /// Which rust pattern archetype this cell follows.
    pub archetype_id: u8,
    /// Sparse delta from archetype (index, value pairs).
    pub delta: SmallVec<[(u8, f32); 4]>,
    /// Accumulated activity history (memory trace).
    pub memory_trace: f32,
    /// Current visual intensity (0.0 - 1.0).
    pub intensity: f32,
    /// Physarum pheromone contribution.
    pub pheromone: f32,
    /// Reaction-diffusion contribution.
    pub reaction: f32,
    /// Heat from database activity.
    pub heat: f32,
    /// Final composed character.
    pub character: char,
    /// Final composed RGB color.
    pub color: (u8, u8, u8),
}

impl Default for TroCell {
    fn default() -> Self {
        Self {
            archetype_id: 0,
            delta: SmallVec::new(),
            memory_trace: 0.0,
            intensity: 0.0,
            pheromone: 0.0,
            reaction: 0.0,
            heat: 0.0,
            character: ' ',
            color: (0, 17, 0), // Dark green base
        }
    }
}

impl TroCell {
    /// Creates a new cell with the given archetype.
    #[must_use]
    pub fn with_archetype(archetype_id: u8) -> Self {
        Self {
            archetype_id,
            ..Self::default()
        }
    }

    /// Computes the residual (difference from stable state).
    #[must_use]
    pub fn residual(&self) -> f32 {
        (self.intensity - self.memory_trace).abs()
    }

    /// Updates the memory trace towards current intensity.
    pub fn update_memory(&mut self, learning_rate: f32) {
        self.memory_trace += (self.intensity - self.memory_trace) * learning_rate;
    }

    /// Decays heat over time.
    pub fn decay_heat(&mut self, decay_rate: f32) {
        self.heat *= decay_rate;
    }
}

/// The complete TRO organism state.
pub struct TroState {
    /// Width of the border area.
    pub width: u16,
    /// Height of the border area.
    pub height: u16,
    /// Total border length (perimeter).
    pub border_length: usize,
    /// Cells around the border (linearized perimeter).
    pub cells: Vec<TroCell>,
    /// Physarum slime mold simulation.
    pub physarum: PhysarumField,
    /// Reaction-diffusion field.
    pub reaction: ReactionField,
    /// Archetype registry for pattern recognition.
    pub archetypes: ArchetypeRegistry,
    /// Current color palette.
    pub palette: Palette,
    /// Frame counter for timing effects.
    pub frame: u64,
    /// Character set mode (Unicode or ASCII).
    pub charset_mode: CharsetMode,
    /// Resolution mode for sub-pixel rendering.
    pub resolution: ResolutionMode,
    /// Border depth in cell layers (1-5).
    pub border_depth: u8,
    /// 2D sparse border region for enhanced rendering.
    pub border_region: BorderRegion,
    /// Tendril manager for organic growth effects.
    pub tendril_manager: TendrilManager,
}

impl TroState {
    /// Creates a new TRO state with given dimensions.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // i % 8 always fits in u8
    pub fn new(width: u16, height: u16, agent_count: usize, palette: Palette) -> Self {
        Self::with_config(
            width,
            height,
            agent_count,
            palette,
            TendrilConfig::default(),
        )
    }

    /// Creates a new TRO state with custom tendril configuration.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // i % 8 always fits in u8
    pub fn with_config(
        width: u16,
        height: u16,
        agent_count: usize,
        palette: Palette,
        tendril_config: TendrilConfig,
    ) -> Self {
        let border_length = Self::compute_border_length(width, height);
        let border_depth: u8 = 1; // Single-cell border on perimeter only

        let cells = (0..border_length)
            .map(|i| {
                let archetype_id = (i % 8) as u8;
                TroCell::with_archetype(archetype_id)
            })
            .collect();

        let physarum = PhysarumField::new(border_length, agent_count);
        let reaction = ReactionField::new(border_length);
        let archetypes = ArchetypeRegistry::default();

        // Create 2D border region with configurable depth
        let border_region = BorderRegion::new(width, height, border_depth);

        // Create tendril manager
        let tendril_manager = TendrilManager::new(tendril_config, width, height);

        Self {
            width,
            height,
            border_length,
            cells,
            physarum,
            reaction,
            archetypes,
            palette,
            frame: 0,
            charset_mode: CharsetMode::default(),
            resolution: ResolutionMode::default(),
            border_depth,
            border_region,
            tendril_manager,
        }
    }

    /// Computes the border length for given dimensions.
    fn compute_border_length(width: u16, height: u16) -> usize {
        if width < 2 || height < 2 {
            return 0;
        }
        // Perimeter: top + bottom + left + right (minus corners counted twice)
        2 * (width as usize) + 2 * (height as usize).saturating_sub(2)
    }

    /// Converts a linear border index to (x, y) coordinates.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Border coords fit in u16
    pub fn index_to_coord(&self, index: usize) -> (u16, u16) {
        let w = self.width as usize;
        let h = self.height as usize;

        if w < 2 || h < 2 {
            return (0, 0);
        }

        let top_len = w;
        let right_len = h.saturating_sub(2);
        let bottom_len = w;

        if index < top_len {
            // Top edge (left to right)
            (index as u16, 0)
        } else if index < top_len + right_len {
            // Right edge (top to bottom, excluding corners)
            let offset = index - top_len;
            (w as u16 - 1, (offset + 1) as u16)
        } else if index < top_len + right_len + bottom_len {
            // Bottom edge (right to left)
            let offset = index - top_len - right_len;
            ((w - 1 - offset) as u16, h as u16 - 1)
        } else {
            // Left edge (bottom to top, excluding corners)
            let offset = index - top_len - right_len - bottom_len;
            (0, (h - 2 - offset) as u16)
        }
    }

    /// Converts (x, y) coordinates to a linear border index.
    #[must_use]
    pub fn coord_to_index(&self, x: u16, y: u16) -> Option<usize> {
        let w = self.width;
        let h = self.height;

        if w < 2 || h < 2 {
            return None;
        }

        // Check if coordinate is on the border
        if y == 0 {
            // Top edge
            Some(x as usize)
        } else if x == w - 1 && y > 0 && y < h - 1 {
            // Right edge
            Some(w as usize + (y - 1) as usize)
        } else if y == h - 1 {
            // Bottom edge
            Some(w as usize + (h - 2) as usize + (w - 1 - x) as usize)
        } else if x == 0 && y > 0 && y < h - 1 {
            // Left edge
            Some(w as usize + (h - 2) as usize + w as usize + (h - 2 - y) as usize)
        } else {
            None
        }
    }

    /// Injects a pulse of activity at a border position.
    pub fn inject_pulse(&mut self, position: usize, intensity: f32) {
        if position < self.cells.len() {
            self.cells[position].heat += intensity;
            // Also spawn some agents near the pulse
            self.physarum.spawn_agents_at(position, 5, intensity);
        }
    }

    /// Applies activity heat from the sensor to cells.
    pub fn apply_activity_heat(&mut self, heat: &[f32]) {
        for (cell, &h) in self.cells.iter_mut().zip(heat.iter()) {
            cell.heat += h * 0.1;
        }
    }

    /// Composes all layers into final cell values.
    pub fn compose_layers(&mut self) {
        let palette_colors = self.palette.get_colors();
        let cell_count = self.cells.len();

        // First pass: gather pheromone and reaction values
        let pheromone_values: Vec<f32> = (0..cell_count)
            .map(|i| self.physarum.get_pheromone(i))
            .collect();
        let reaction_values: Vec<f32> = (0..cell_count)
            .map(|i| self.reaction.get_value(i))
            .collect();

        // Second pass: update cells
        for (i, cell) in self.cells.iter_mut().enumerate() {
            let pheromone = pheromone_values[i];
            let reaction = reaction_values[i];

            // Store raw values
            cell.pheromone = pheromone;
            cell.reaction = reaction;

            // Compose intensity: pheromone is primary, reaction adds depth
            let base_intensity = pheromone * 0.7 + reaction * 0.2;

            // Add heat contribution (database activity)
            let heat_boost = cell.heat * 0.5;
            cell.intensity = (base_intensity + heat_boost).clamp(0.0, 1.0);

            // Update memory trace (slow learning)
            cell.update_memory(0.02);

            // Decay heat
            cell.decay_heat(0.95);

            // Select character based on intensity and dominant layer
            cell.character = Self::select_character_static(pheromone, reaction, cell.heat);

            // Select color based on intensity and heat
            cell.color = Self::interpolate_color_static(cell.intensity, cell.heat, palette_colors);
        }

        self.frame = self.frame.wrapping_add(1);
    }

    /// Selects the appropriate character based on layer contributions (static version).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn select_character_static(pheromone: f32, reaction: f32, heat: f32) -> char {
        use super::charset::{HEAT_CHARS, ORGANIC_CHARS, RUST_CHARS, TRAIL_CHARS};

        let total = pheromone + reaction + heat;
        if total < 0.05 {
            return ' ';
        }

        // Determine dominant layer
        let chars = if heat > pheromone && heat > reaction {
            &HEAT_CHARS
        } else if pheromone > reaction {
            &TRAIL_CHARS
        } else if reaction > 0.3 {
            &ORGANIC_CHARS
        } else {
            &RUST_CHARS
        };

        // Map intensity to character index
        let intensity = total.clamp(0.0, 1.0);
        let index = ((intensity * (chars.len() - 1) as f32) as usize).min(chars.len() - 1);
        chars[index]
    }

    /// Interpolates color based on intensity and heat (static version).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn interpolate_color_static(
        intensity: f32,
        heat: f32,
        palette: &[(u8, u8, u8)],
    ) -> (u8, u8, u8) {
        if palette.is_empty() {
            return (0, 51, 0);
        }

        // Map intensity to palette position
        let base_idx = ((intensity * (palette.len() - 1) as f32) as usize).min(palette.len() - 1);
        let base_color = palette[base_idx];

        // Add red tint for heat (database activity)
        if heat > 0.1 {
            let heat_factor = (heat * 0.5).min(0.3);
            let r = (base_color.0 as f32 + heat_factor * 200.0).min(255.0) as u8;
            let g = (base_color.1 as f32 * (1.0 - heat_factor * 0.3)).max(0.0) as u8;
            let b = base_color.2;
            (r, g, b)
        } else {
            base_color
        }
    }

    /// Checks if the organism has converged (90% of cells stable).
    #[must_use]
    pub fn is_converged(&self) -> bool {
        let stable_count = self.cells.iter().filter(|c| c.residual() < 0.01).count();
        stable_count as f32 / self.cells.len() as f32 >= 0.9
    }

    /// Resizes the TRO state to new dimensions.
    pub fn resize(&mut self, width: u16, height: u16) {
        let new_length = Self::compute_border_length(width, height);

        if new_length != self.border_length || self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.border_length = new_length;

            // Resize cells
            self.cells.resize_with(new_length, TroCell::default);

            // Resize fields
            self.physarum.resize(new_length);
            self.reaction.resize(new_length);

            // Resize 2D border region and tendril manager
            self.border_region.resize(width, height);
            self.tendril_manager.resize(width, height);
        }
    }

    /// Gets the cell at a given border index.
    #[must_use]
    pub fn get_cell(&self, index: usize) -> Option<&TroCell> {
        self.cells.get(index)
    }

    /// Updates the 2D border region from simulation data.
    pub fn update_border_region(&mut self) {
        // Collect pheromone data for border cells
        let pheromone_data: Vec<f32> = (0..self.border_length)
            .map(|i| self.physarum.get_pheromone(i))
            .collect();

        // Collect heat data
        let heat_data: Vec<f32> = self.cells.iter().map(|c| c.heat).collect();

        // Update border region from physarum data
        self.border_region
            .update_from_linear(&pheromone_data, &heat_data, self.width, self.height);

        // Update tendrils based on pheromone and heat
        self.tendril_manager.update(&pheromone_data, &heat_data);
    }

    /// Sets the border depth and rebuilds the region.
    pub fn set_border_depth(&mut self, depth: u8) {
        self.border_depth = depth.clamp(1, 5);
        self.border_region = BorderRegion::new(self.width, self.height, self.border_depth);
    }

    /// Sets the resolution mode.
    pub fn set_resolution(&mut self, mode: ResolutionMode) {
        self.resolution = mode;
        self.border_region.set_resolution(mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_default() {
        let cell = TroCell::default();
        assert_eq!(cell.archetype_id, 0);
        assert_eq!(cell.intensity, 0.0);
        assert_eq!(cell.character, ' ');
    }

    #[test]
    fn test_cell_with_archetype() {
        let cell = TroCell::with_archetype(5);
        assert_eq!(cell.archetype_id, 5);
    }

    #[test]
    fn test_cell_residual() {
        let mut cell = TroCell::default();
        cell.intensity = 0.5;
        cell.memory_trace = 0.3;
        assert!((cell.residual() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_cell_decay_heat() {
        let mut cell = TroCell::default();
        cell.heat = 1.0;
        cell.decay_heat(0.5);
        assert!((cell.heat - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_state_border_length() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        // Perimeter: 2*80 + 2*(24-2) = 160 + 44 = 204
        assert_eq!(state.border_length, 204);
    }

    #[test]
    fn test_state_border_length_small() {
        // Minimum case
        let state = TroState::new(2, 2, 10, Palette::PhosphorGreen);
        // Perimeter: 2*2 + 2*0 = 4
        assert_eq!(state.border_length, 4);
    }

    #[test]
    fn test_state_border_length_invalid() {
        let state = TroState::new(1, 1, 10, Palette::PhosphorGreen);
        assert_eq!(state.border_length, 0);
    }

    #[test]
    fn test_index_to_coord_top() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        assert_eq!(state.index_to_coord(0), (0, 0));
        assert_eq!(state.index_to_coord(79), (79, 0));
    }

    #[test]
    fn test_index_to_coord_right() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        assert_eq!(state.index_to_coord(80), (79, 1));
        assert_eq!(state.index_to_coord(81), (79, 2));
    }

    #[test]
    fn test_index_to_coord_bottom() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        // After top (80) and right side (22), bottom starts at 102
        assert_eq!(state.index_to_coord(102), (79, 23));
        assert_eq!(state.index_to_coord(103), (78, 23));
    }

    #[test]
    fn test_coord_to_index_roundtrip() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        for i in 0..state.border_length {
            let (x, y) = state.index_to_coord(i);
            let idx = state.coord_to_index(x, y);
            assert_eq!(idx, Some(i), "Failed at index {i}, coord ({x}, {y})");
        }
    }

    #[test]
    fn test_inject_pulse() {
        let mut state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        state.inject_pulse(10, 0.5);
        assert!(state.cells[10].heat > 0.0);
    }

    #[test]
    fn test_resize() {
        let mut state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        state.resize(100, 30);
        assert_eq!(state.width, 100);
        assert_eq!(state.height, 30);
        let expected_length = 2 * 100 + 2 * (30 - 2);
        assert_eq!(state.border_length, expected_length);
    }

    #[test]
    fn test_is_converged_initial() {
        let state = TroState::new(80, 24, 100, Palette::PhosphorGreen);
        // Initially all cells have intensity == memory_trace == 0, so converged
        assert!(state.is_converged());
    }
}
