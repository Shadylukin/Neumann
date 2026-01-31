// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2D sparse border region for multi-cell thick organic mold patterns.
//!
//! Replaces the 1D linearized perimeter with a 2D region that supports
//! variable border depth and sub-pixel rendering resolutions.

// Visual calculations involve many float/int conversions with acceptable precision loss.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;

use smallvec::SmallVec;

/// 2D coordinate in the border region.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct BorderCoord {
    pub x: u16,
    pub y: u16,
}

impl BorderCoord {
    /// Creates a new border coordinate.
    #[must_use]
    pub const fn new(x: u16, y: u16) -> Self {
        Self { x, y }
    }

    /// Calculates Manhattan distance to another coordinate.
    #[must_use]
    pub fn manhattan_distance(&self, other: &Self) -> u32 {
        let dx = (i32::from(self.x) - i32::from(other.x)).unsigned_abs();
        let dy = (i32::from(self.y) - i32::from(other.y)).unsigned_abs();
        dx + dy
    }
}

/// Resolution mode for sub-cell rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResolutionMode {
    /// Standard 1x1 character resolution.
    #[default]
    Standard,
    /// Half-block 1x2 vertical resolution.
    HalfBlock,
    /// Quadrant 2x2 resolution.
    Quadrant,
    /// Braille 2x4 resolution.
    Braille,
}

impl ResolutionMode {
    /// Returns the number of sub-pixels per cell.
    #[must_use]
    pub const fn sub_pixel_count(&self) -> usize {
        match self {
            Self::Standard => 1,
            Self::HalfBlock => 2,
            Self::Quadrant => 4,
            Self::Braille => 8,
        }
    }

    /// Parses resolution mode from string.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "standard" | "1x1" | "normal" => Some(Self::Standard),
            "half" | "halfblock" | "half-block" | "1x2" => Some(Self::HalfBlock),
            "quad" | "quadrant" | "2x2" => Some(Self::Quadrant),
            "braille" | "2x4" => Some(Self::Braille),
            _ => None,
        }
    }

    /// Returns the name of this resolution mode.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::HalfBlock => "halfblock",
            Self::Quadrant => "quadrant",
            Self::Braille => "braille",
        }
    }
}

/// Border cell with sub-pixel intensities.
#[derive(Debug, Clone)]
pub struct BorderCell {
    /// Sub-pixel intensities (1, 2, 4, or 8 values depending on resolution).
    pub intensities: SmallVec<[f32; 8]>,
    /// Primary intensity for color selection (average of sub-pixels).
    pub primary_intensity: f32,
    /// Heat contribution from database activity.
    pub heat: f32,
    /// Distance from outer edge (0 = perimeter).
    pub depth: u8,
    /// Pheromone level from physarum simulation.
    pub pheromone: f32,
    /// Reaction-diffusion contribution.
    pub reaction: f32,
}

impl Default for BorderCell {
    fn default() -> Self {
        Self {
            intensities: SmallVec::from_elem(0.0, 1),
            primary_intensity: 0.0,
            heat: 0.0,
            depth: 0,
            pheromone: 0.0,
            reaction: 0.0,
        }
    }
}

impl BorderCell {
    /// Creates a new cell with the given depth and resolution.
    #[must_use]
    pub fn new(depth: u8, resolution: ResolutionMode) -> Self {
        Self {
            intensities: SmallVec::from_elem(0.0, resolution.sub_pixel_count()),
            depth,
            ..Self::default()
        }
    }

    /// Updates primary intensity from sub-pixel values.
    pub fn update_primary(&mut self) {
        if self.intensities.is_empty() {
            self.primary_intensity = 0.0;
        } else {
            let sum: f32 = self.intensities.iter().sum();
            self.primary_intensity = sum / self.intensities.len() as f32;
        }
    }

    /// Sets all sub-pixels to a uniform intensity.
    pub fn set_uniform(&mut self, intensity: f32) {
        for v in &mut self.intensities {
            *v = intensity;
        }
        self.primary_intensity = intensity;
    }

    /// Decays the cell values over time.
    pub fn decay(&mut self, rate: f32) {
        for v in &mut self.intensities {
            *v *= rate;
        }
        self.heat *= rate;
        self.pheromone *= rate;
        self.update_primary();
    }
}

/// Sparse 2D border region.
pub struct BorderRegion {
    width: u16,
    height: u16,
    max_depth: u8,
    resolution: ResolutionMode,
    cells: HashMap<BorderCoord, BorderCell>,
}

impl BorderRegion {
    /// Creates a new border region with the given dimensions and depth.
    #[must_use]
    pub fn new(width: u16, height: u16, max_depth: u8) -> Self {
        let mut region = Self {
            width,
            height,
            max_depth,
            resolution: ResolutionMode::default(),
            cells: HashMap::new(),
        };
        region.initialize_cells();
        region
    }

    /// Initializes cells for the border region.
    fn initialize_cells(&mut self) {
        self.cells.clear();

        if self.width < 2 || self.height < 2 {
            return;
        }

        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(depth) = self.compute_depth(x, y) {
                    let coord = BorderCoord::new(x, y);
                    let cell = BorderCell::new(depth, self.resolution);
                    self.cells.insert(coord, cell);
                }
            }
        }
    }

    /// Computes the depth of a coordinate from the edge.
    /// Returns None if the coordinate is not in the border region.
    fn compute_depth(&self, x: u16, y: u16) -> Option<u8> {
        let dist_left = x;
        let dist_right = self.width.saturating_sub(1).saturating_sub(x);
        let dist_top = y;
        let dist_bottom = self.height.saturating_sub(1).saturating_sub(y);

        let min_dist = dist_left.min(dist_right).min(dist_top).min(dist_bottom);

        if min_dist < u16::from(self.max_depth) {
            // Depth is clamped to u8 range; max_depth is already u8
            #[allow(clippy::cast_possible_truncation)]
            Some(min_dist as u8)
        } else {
            None
        }
    }

    /// Checks if a coordinate is within the border region.
    #[must_use]
    pub fn is_border(&self, x: u16, y: u16) -> bool {
        self.compute_depth(x, y).is_some()
    }

    /// Returns the width of the region.
    #[must_use]
    pub const fn width(&self) -> u16 {
        self.width
    }

    /// Returns the height of the region.
    #[must_use]
    pub const fn height(&self) -> u16 {
        self.height
    }

    /// Returns the maximum border depth.
    #[must_use]
    pub const fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Returns the current resolution mode.
    #[must_use]
    pub const fn resolution(&self) -> ResolutionMode {
        self.resolution
    }

    /// Sets the resolution mode and reinitializes cells.
    pub fn set_resolution(&mut self, mode: ResolutionMode) {
        if self.resolution != mode {
            self.resolution = mode;
            // Update existing cells to new resolution
            for cell in self.cells.values_mut() {
                let old_primary = cell.primary_intensity;
                cell.intensities = SmallVec::from_elem(old_primary, mode.sub_pixel_count());
            }
        }
    }

    /// Sets the maximum border depth and reinitializes cells.
    pub fn set_max_depth(&mut self, depth: u8) {
        if self.max_depth != depth {
            self.max_depth = depth;
            self.initialize_cells();
        }
    }

    /// Resizes the region to new dimensions.
    pub fn resize(&mut self, width: u16, height: u16) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.initialize_cells();
        }
    }

    /// Returns the number of active cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Returns a reference to a cell at the given coordinate.
    #[must_use]
    pub fn get(&self, coord: &BorderCoord) -> Option<&BorderCell> {
        self.cells.get(coord)
    }

    /// Returns a mutable reference to a cell at the given coordinate.
    pub fn get_mut(&mut self, coord: &BorderCoord) -> Option<&mut BorderCell> {
        self.cells.get_mut(coord)
    }

    /// Iterates over all active cells.
    pub fn iter(&self) -> impl Iterator<Item = (&BorderCoord, &BorderCell)> {
        self.cells.iter()
    }

    /// Iterates over all active cells mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&BorderCoord, &mut BorderCell)> {
        self.cells.iter_mut()
    }

    /// Returns cells at a specific depth.
    pub fn cells_at_depth(&self, depth: u8) -> impl Iterator<Item = (&BorderCoord, &BorderCell)> {
        self.cells
            .iter()
            .filter(move |(_, cell)| cell.depth == depth)
    }

    /// Decays all cell values.
    pub fn decay_all(&mut self, rate: f32) {
        for cell in self.cells.values_mut() {
            cell.decay(rate);
        }
    }

    /// Injects heat at a coordinate with falloff.
    pub fn inject_heat(&mut self, coord: &BorderCoord, intensity: f32, falloff: f32) {
        for (c, cell) in &mut self.cells {
            let dist = coord.manhattan_distance(c);
            let factor = (-falloff * dist as f32).exp();
            cell.heat += intensity * factor;
        }
    }

    /// Converts a linear border index to 2D coordinate (for compatibility).
    /// Walks the perimeter: top -> right -> bottom -> left.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn linear_to_coord(&self, index: usize) -> Option<BorderCoord> {
        let w = self.width as usize;
        let h = self.height as usize;

        if w < 2 || h < 2 {
            return None;
        }

        let top_len = w;
        let right_len = h.saturating_sub(2);
        let bottom_len = w;
        let left_len = h.saturating_sub(2);
        let total = top_len + right_len + bottom_len + left_len;

        if index >= total {
            return None;
        }

        if index < top_len {
            Some(BorderCoord::new(index as u16, 0))
        } else if index < top_len + right_len {
            let offset = index - top_len;
            Some(BorderCoord::new(self.width - 1, (offset + 1) as u16))
        } else if index < top_len + right_len + bottom_len {
            let offset = index - top_len - right_len;
            Some(BorderCoord::new((w - 1 - offset) as u16, self.height - 1))
        } else {
            let offset = index - top_len - right_len - bottom_len;
            Some(BorderCoord::new(0, (h - 2 - offset) as u16))
        }
    }

    /// Converts a 2D coordinate to linear border index (perimeter only).
    #[must_use]
    pub fn coord_to_linear(&self, coord: &BorderCoord) -> Option<usize> {
        let w = self.width;
        let h = self.height;
        let x = coord.x;
        let y = coord.y;

        if w < 2 || h < 2 {
            return None;
        }

        if y == 0 {
            Some(x as usize)
        } else if x == w - 1 && y > 0 && y < h - 1 {
            Some(w as usize + (y - 1) as usize)
        } else if y == h - 1 {
            Some(w as usize + (h - 2) as usize + (w - 1 - x) as usize)
        } else if x == 0 && y > 0 && y < h - 1 {
            Some(w as usize + (h - 2) as usize + w as usize + (h - 2 - y) as usize)
        } else {
            None
        }
    }

    /// Updates border region cells from 1D linearized pheromone and heat data.
    /// This propagates values from the perimeter into the inner depth layers.
    pub fn update_from_linear(
        &mut self,
        pheromone: &[f32],
        heat: &[f32],
        _width: u16,
        _height: u16,
    ) {
        // First pass: collect perimeter cell updates
        let perimeter_updates: Vec<(BorderCoord, f32, f32)> = self
            .cells
            .iter()
            .filter(|(_, c)| c.depth == 0)
            .filter_map(|(coord, _)| {
                Self::coord_to_linear_static(*coord, self.width, self.height).map(|idx| {
                    let p = pheromone.get(idx).copied().unwrap_or(0.0);
                    let h = heat.get(idx).copied().unwrap_or(0.0);
                    (*coord, p, h)
                })
            })
            .collect();

        // Apply perimeter updates
        for (coord, p, h) in &perimeter_updates {
            if let Some(cell) = self.cells.get_mut(coord) {
                cell.pheromone = *p;
                cell.set_uniform(*p);
                cell.heat = *h;
            }
        }

        // Collect inner cell coordinates and their depths
        let inner_coords: Vec<(BorderCoord, u8)> = self
            .cells
            .iter()
            .filter(|(_, c)| c.depth > 0)
            .map(|(coord, cell)| (*coord, cell.depth))
            .collect();

        // Update inner cells based on nearest perimeter cells
        for (coord, depth) in inner_coords {
            let mut total_intensity = 0.0;
            let mut total_heat = 0.0;
            let mut weight_sum = 0.0;

            for (perim_coord, intensity, heat_val) in &perimeter_updates {
                let dist = coord.manhattan_distance(perim_coord) as f32;
                let weight = (-dist * 0.3).exp();
                total_intensity += intensity * weight;
                total_heat += heat_val * weight;
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                let avg_intensity = total_intensity / weight_sum;
                let avg_heat = total_heat / weight_sum;

                // Apply depth-based falloff
                let depth_factor = f32::from(depth).mul_add(-0.15, 1.0);

                if let Some(cell) = self.cells.get_mut(&coord) {
                    cell.set_uniform(avg_intensity * depth_factor);
                    cell.heat = avg_heat * depth_factor;
                }
            }
        }
    }

    /// Static version of `coord_to_linear` for use in borrow contexts.
    fn coord_to_linear_static(coord: BorderCoord, width: u16, height: u16) -> Option<usize> {
        let w = width;
        let h = height;
        let x = coord.x;
        let y = coord.y;

        if w < 2 || h < 2 {
            return None;
        }

        if y == 0 {
            Some(x as usize)
        } else if x == w - 1 && y > 0 && y < h - 1 {
            Some(w as usize + (y - 1) as usize)
        } else if y == h - 1 {
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
    fn test_border_coord_new() {
        let coord = BorderCoord::new(10, 20);
        assert_eq!(coord.x, 10);
        assert_eq!(coord.y, 20);
    }

    #[test]
    fn test_border_coord_manhattan() {
        let a = BorderCoord::new(0, 0);
        let b = BorderCoord::new(3, 4);
        assert_eq!(a.manhattan_distance(&b), 7);
    }

    #[test]
    fn test_resolution_mode_default() {
        assert_eq!(ResolutionMode::default(), ResolutionMode::Standard);
    }

    #[test]
    fn test_resolution_mode_sub_pixel_count() {
        assert_eq!(ResolutionMode::Standard.sub_pixel_count(), 1);
        assert_eq!(ResolutionMode::HalfBlock.sub_pixel_count(), 2);
        assert_eq!(ResolutionMode::Quadrant.sub_pixel_count(), 4);
        assert_eq!(ResolutionMode::Braille.sub_pixel_count(), 8);
    }

    #[test]
    fn test_resolution_mode_from_name() {
        assert_eq!(
            ResolutionMode::from_name("braille"),
            Some(ResolutionMode::Braille)
        );
        assert_eq!(
            ResolutionMode::from_name("quadrant"),
            Some(ResolutionMode::Quadrant)
        );
        assert_eq!(ResolutionMode::from_name("invalid"), None);
    }

    #[test]
    fn test_resolution_mode_name() {
        assert_eq!(ResolutionMode::Standard.name(), "standard");
        assert_eq!(ResolutionMode::Braille.name(), "braille");
    }

    #[test]
    fn test_border_cell_default() {
        let cell = BorderCell::default();
        assert_eq!(cell.primary_intensity, 0.0);
        assert_eq!(cell.depth, 0);
    }

    #[test]
    fn test_border_cell_new() {
        let cell = BorderCell::new(2, ResolutionMode::Quadrant);
        assert_eq!(cell.depth, 2);
        assert_eq!(cell.intensities.len(), 4);
    }

    #[test]
    fn test_border_cell_set_uniform() {
        let mut cell = BorderCell::new(0, ResolutionMode::Braille);
        cell.set_uniform(0.5);
        assert_eq!(cell.primary_intensity, 0.5);
        for &v in &cell.intensities {
            assert!((v - 0.5).abs() < 0.001);
        }
    }

    #[test]
    fn test_border_cell_decay() {
        let mut cell = BorderCell::new(0, ResolutionMode::Standard);
        cell.set_uniform(1.0);
        cell.heat = 1.0;
        cell.decay(0.5);
        assert!((cell.primary_intensity - 0.5).abs() < 0.001);
        assert!((cell.heat - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_border_region_new() {
        let region = BorderRegion::new(80, 24, 3);
        assert_eq!(region.width(), 80);
        assert_eq!(region.height(), 24);
        assert_eq!(region.max_depth(), 3);
    }

    #[test]
    fn test_border_region_is_border() {
        let region = BorderRegion::new(80, 24, 2);

        // Corners should be border
        assert!(region.is_border(0, 0));
        assert!(region.is_border(79, 0));
        assert!(region.is_border(0, 23));
        assert!(region.is_border(79, 23));

        // Edge at depth 1 should be border
        assert!(region.is_border(1, 0));
        assert!(region.is_border(0, 1));

        // Center should not be border
        assert!(!region.is_border(40, 12));
    }

    #[test]
    fn test_border_region_cell_count() {
        let region = BorderRegion::new(10, 10, 1);
        // Perimeter only: 10*4 - 4 (corners) = 36
        assert_eq!(region.cell_count(), 36);

        let region2 = BorderRegion::new(10, 10, 2);
        // Two layers: perimeter + inner ring
        // Outer: 36, Inner: 8*4 - 4 = 28
        assert_eq!(region2.cell_count(), 64);
    }

    #[test]
    fn test_border_region_get() {
        let region = BorderRegion::new(80, 24, 1);
        let coord = BorderCoord::new(0, 0);
        assert!(region.get(&coord).is_some());

        let inner = BorderCoord::new(40, 12);
        assert!(region.get(&inner).is_none());
    }

    #[test]
    fn test_border_region_set_resolution() {
        let mut region = BorderRegion::new(10, 10, 1);
        region.set_resolution(ResolutionMode::Braille);
        assert_eq!(region.resolution(), ResolutionMode::Braille);

        // Check cells have correct intensity count
        for (_, cell) in region.iter() {
            assert_eq!(cell.intensities.len(), 8);
        }
    }

    #[test]
    fn test_border_region_resize() {
        let mut region = BorderRegion::new(10, 10, 1);
        let initial_count = region.cell_count();

        region.resize(20, 20);
        assert_eq!(region.width(), 20);
        assert_eq!(region.height(), 20);
        assert!(region.cell_count() > initial_count);
    }

    #[test]
    fn test_border_region_linear_to_coord() {
        let region = BorderRegion::new(80, 24, 1);

        // Top edge
        assert_eq!(region.linear_to_coord(0), Some(BorderCoord::new(0, 0)));
        assert_eq!(region.linear_to_coord(79), Some(BorderCoord::new(79, 0)));

        // Right edge
        assert_eq!(region.linear_to_coord(80), Some(BorderCoord::new(79, 1)));
    }

    #[test]
    fn test_border_region_coord_to_linear() {
        let region = BorderRegion::new(80, 24, 1);

        let coord = BorderCoord::new(0, 0);
        assert_eq!(region.coord_to_linear(&coord), Some(0));

        let coord2 = BorderCoord::new(79, 0);
        assert_eq!(region.coord_to_linear(&coord2), Some(79));

        // Non-border coordinate
        let inner = BorderCoord::new(40, 12);
        assert_eq!(region.coord_to_linear(&inner), None);
    }

    #[test]
    fn test_border_region_roundtrip() {
        let region = BorderRegion::new(80, 24, 1);
        let perimeter = 2 * 80 + 2 * (24 - 2);

        for i in 0..perimeter {
            let coord = region.linear_to_coord(i).expect("valid index");
            let back = region.coord_to_linear(&coord).expect("valid coord");
            assert_eq!(i, back, "Roundtrip failed at index {i}");
        }
    }

    #[test]
    fn test_border_region_cells_at_depth() {
        let region = BorderRegion::new(10, 10, 2);

        let depth_0: Vec<_> = region.cells_at_depth(0).collect();
        let depth_1: Vec<_> = region.cells_at_depth(1).collect();

        // All collected cells should have the correct depth
        for (_, cell) in &depth_0 {
            assert_eq!(cell.depth, 0);
        }
        for (_, cell) in &depth_1 {
            assert_eq!(cell.depth, 1);
        }
    }

    #[test]
    fn test_border_region_inject_heat() {
        let mut region = BorderRegion::new(10, 10, 1);
        let center = BorderCoord::new(5, 0);
        region.inject_heat(&center, 1.0, 0.5);

        // The injection point should have heat
        if let Some(cell) = region.get(&center) {
            assert!(cell.heat > 0.0);
        }
    }

    #[test]
    fn test_border_region_empty() {
        let region = BorderRegion::new(1, 1, 1);
        assert_eq!(region.cell_count(), 0);
    }
}
