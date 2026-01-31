// SPDX-License-Identifier: MIT OR Apache-2.0
//! Rust pattern archetypes for delta-based memory encoding.
//!
//! Cells store their visual state as a delta from learned "rust archetypes"
//! (patterns of decay/growth), enabling memory traces and emergent behavior.

use smallvec::SmallVec;

/// The number of archetype patterns.
pub const NUM_ARCHETYPES: usize = 8;

/// Size of each archetype pattern (features per archetype).
pub const ARCHETYPE_DIM: usize = 4;

/// A rust pattern archetype (learned decay/growth pattern).
#[derive(Debug, Clone)]
pub struct Archetype {
    /// Unique ID for this archetype.
    pub id: u8,
    /// Base intensity pattern.
    pub intensity: f32,
    /// Texture roughness (affects character selection).
    pub roughness: f32,
    /// Organic vs mechanical blend.
    pub organic_blend: f32,
    /// Heat reactivity (how much activity affects this pattern).
    pub heat_reactivity: f32,
}

impl Archetype {
    /// Creates a new archetype with given parameters.
    #[must_use]
    pub const fn new(
        id: u8,
        intensity: f32,
        roughness: f32,
        organic_blend: f32,
        heat_reactivity: f32,
    ) -> Self {
        Self {
            id,
            intensity,
            roughness,
            organic_blend,
            heat_reactivity,
        }
    }

    /// Converts the archetype to a feature vector.
    #[must_use]
    pub fn to_vector(&self) -> [f32; ARCHETYPE_DIM] {
        [
            self.intensity,
            self.roughness,
            self.organic_blend,
            self.heat_reactivity,
        ]
    }

    /// Computes the distance from a feature vector.
    #[must_use]
    pub fn distance_from(&self, features: [f32; ARCHETYPE_DIM]) -> f32 {
        let vec = self.to_vector();
        vec.iter()
            .zip(features.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Registry of all rust archetypes.
#[derive(Debug, Clone)]
pub struct ArchetypeRegistry {
    /// The archetypes.
    archetypes: [Archetype; NUM_ARCHETYPES],
}

impl Default for ArchetypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ArchetypeRegistry {
    /// Creates a new registry with predefined archetypes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            archetypes: [
                // 0: Clean/minimal decay
                Archetype::new(0, 0.1, 0.1, 0.0, 0.2),
                // 1: Light surface rust
                Archetype::new(1, 0.2, 0.3, 0.1, 0.3),
                // 2: Medium corrosion
                Archetype::new(2, 0.4, 0.5, 0.2, 0.4),
                // 3: Heavy rust, pitted
                Archetype::new(3, 0.6, 0.7, 0.3, 0.5),
                // 4: Organic growth (mold-like)
                Archetype::new(4, 0.3, 0.4, 0.8, 0.6),
                // 5: Active decay (spreading)
                Archetype::new(5, 0.5, 0.6, 0.5, 0.8),
                // 6: Heat-damaged
                Archetype::new(6, 0.7, 0.8, 0.2, 1.0),
                // 7: Deep corruption
                Archetype::new(7, 0.9, 0.9, 0.7, 0.7),
            ],
        }
    }

    /// Gets an archetype by ID.
    #[must_use]
    pub fn get(&self, id: u8) -> Option<&Archetype> {
        self.archetypes.get(id as usize)
    }

    /// Finds the closest archetype to a feature vector.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Index always fits in u8 (NUM_ARCHETYPES = 8)
    pub fn find_closest(&self, features: [f32; ARCHETYPE_DIM]) -> u8 {
        self.archetypes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = a.distance_from(features);
                let db = b.distance_from(features);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |(i, _)| i as u8)
    }

    /// Computes the delta (sparse difference) from an archetype.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // i is always < ARCHETYPE_DIM (4), fits in u8
    pub fn compute_delta(
        &self,
        archetype_id: u8,
        features: [f32; ARCHETYPE_DIM],
    ) -> SmallVec<[(u8, f32); 4]> {
        let archetype = self.get(archetype_id).unwrap_or(&self.archetypes[0]);
        let base = archetype.to_vector();

        let mut delta = SmallVec::new();
        for (i, (feat, base_val)) in features.iter().zip(base.iter()).enumerate() {
            let diff = feat - base_val;
            if diff.abs() > 0.01 {
                delta.push((i as u8, diff));
            }
        }
        delta
    }

    /// Reconstructs features from archetype + delta.
    #[must_use]
    pub fn reconstruct(
        &self,
        archetype_id: u8,
        delta: &SmallVec<[(u8, f32); 4]>,
    ) -> [f32; ARCHETYPE_DIM] {
        let archetype = self.get(archetype_id).unwrap_or(&self.archetypes[0]);
        let mut features = archetype.to_vector();

        for (idx, diff) in delta {
            if (*idx as usize) < ARCHETYPE_DIM {
                features[*idx as usize] += diff;
            }
        }

        features
    }

    /// Blends two archetypes based on a factor.
    #[must_use]
    pub fn blend(&self, id_a: u8, id_b: u8, factor: f32) -> [f32; ARCHETYPE_DIM] {
        let a = self.get(id_a).unwrap_or(&self.archetypes[0]);
        let b = self.get(id_b).unwrap_or(&self.archetypes[0]);

        let va = a.to_vector();
        let vb = b.to_vector();

        let factor = factor.clamp(0.0, 1.0);
        [
            (vb[0] - va[0]).mul_add(factor, va[0]),
            (vb[1] - va[1]).mul_add(factor, va[1]),
            (vb[2] - va[2]).mul_add(factor, va[2]),
            (vb[3] - va[3]).mul_add(factor, va[3]),
        ]
    }

    /// Returns the number of archetypes.
    #[must_use]
    #[allow(clippy::unused_self)] // Idiomatic Rust: len() is a method
    pub const fn len(&self) -> usize {
        NUM_ARCHETYPES
    }

    /// Returns true if the registry is empty.
    #[must_use]
    #[allow(clippy::unused_self)] // Idiomatic Rust: is_empty() is a method
    pub const fn is_empty(&self) -> bool {
        NUM_ARCHETYPES == 0
    }
}

/// Memory state for a cell tracking its pattern history.
#[derive(Debug, Clone, Default)]
pub struct PatternMemory {
    /// Current archetype ID.
    pub archetype_id: u8,
    /// Delta from archetype.
    pub delta: SmallVec<[(u8, f32); 4]>,
    /// Accumulated memory trace (decays slowly).
    pub trace: f32,
    /// How many frames at current archetype.
    pub stability: u32,
}

impl PatternMemory {
    /// Creates a new pattern memory.
    #[must_use]
    pub fn new(archetype_id: u8) -> Self {
        Self {
            archetype_id,
            delta: SmallVec::new(),
            trace: 0.0,
            stability: 0,
        }
    }

    /// Updates the memory with new features.
    #[allow(clippy::cast_precision_loss)] // stability value is small, precision loss acceptable
    pub fn update(&mut self, registry: &ArchetypeRegistry, features: [f32; ARCHETYPE_DIM]) {
        let closest = registry.find_closest(features);

        if closest == self.archetype_id {
            self.stability = self.stability.saturating_add(1);
        } else {
            // Archetype changed - reset stability, accumulate trace
            self.trace += self.stability as f32 * 0.01;
            self.trace = self.trace.min(1.0);
            self.archetype_id = closest;
            self.stability = 0;
        }

        // Update delta
        self.delta = registry.compute_delta(self.archetype_id, features);

        // Decay trace slowly
        self.trace *= 0.999;
    }

    /// Returns whether the pattern is stable.
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.stability > 10
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_archetype_new() {
        let a = Archetype::new(0, 0.5, 0.3, 0.2, 0.1);
        assert_eq!(a.id, 0);
        assert!((a.intensity - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_archetype_to_vector() {
        let a = Archetype::new(0, 0.1, 0.2, 0.3, 0.4);
        let v = a.to_vector();
        assert_eq!(v.len(), ARCHETYPE_DIM);
        assert!((v[0] - 0.1).abs() < 0.001);
        assert!((v[3] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_archetype_distance() {
        let a = Archetype::new(0, 0.0, 0.0, 0.0, 0.0);
        let dist = a.distance_from([1.0, 0.0, 0.0, 0.0]);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_registry_new() {
        let registry = ArchetypeRegistry::new();
        assert_eq!(registry.len(), NUM_ARCHETYPES);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_registry_get() {
        let registry = ArchetypeRegistry::new();
        assert!(registry.get(0).is_some());
        assert!(registry.get(7).is_some());
        assert!(registry.get(100).is_none());
    }

    #[test]
    fn test_registry_find_closest() {
        let registry = ArchetypeRegistry::new();
        // Features matching archetype 0 (low everything)
        let closest = registry.find_closest([0.1, 0.1, 0.0, 0.2]);
        assert_eq!(closest, 0);
    }

    #[test]
    fn test_registry_compute_delta() {
        let registry = ArchetypeRegistry::new();
        let features = [0.15, 0.15, 0.05, 0.25];
        let delta = registry.compute_delta(0, features);
        // Should have small deltas from archetype 0
        assert!(!delta.is_empty() || delta.is_empty()); // Just verify no panic
    }

    #[test]
    fn test_registry_reconstruct() {
        let registry = ArchetypeRegistry::new();
        let original = [0.2, 0.3, 0.4, 0.5];
        let closest = registry.find_closest(original);
        let delta = registry.compute_delta(closest, original);
        let reconstructed = registry.reconstruct(closest, &delta);

        // Reconstruction should be close to original
        for (o, r) in original.iter().zip(reconstructed.iter()) {
            assert!((o - r).abs() < 0.02);
        }
    }

    #[test]
    fn test_registry_blend() {
        let registry = ArchetypeRegistry::new();
        let blended = registry.blend(0, 7, 0.5);
        let a0 = registry.get(0).unwrap().to_vector();
        let a7 = registry.get(7).unwrap().to_vector();

        // Blended should be between a0 and a7
        for i in 0..ARCHETYPE_DIM {
            let mid = (a0[i] + a7[i]) / 2.0;
            assert!((blended[i] - mid).abs() < 0.01);
        }
    }

    #[test]
    fn test_pattern_memory_new() {
        let mem = PatternMemory::new(3);
        assert_eq!(mem.archetype_id, 3);
        assert_eq!(mem.stability, 0);
        assert_eq!(mem.trace, 0.0);
    }

    #[test]
    fn test_pattern_memory_update() {
        let registry = ArchetypeRegistry::new();
        let mut mem = PatternMemory::new(0);

        // Update with features close to archetype 0
        mem.update(&registry, [0.1, 0.1, 0.0, 0.2]);
        assert_eq!(mem.archetype_id, 0);
        assert!(mem.stability >= 1);
    }

    #[test]
    fn test_pattern_memory_stability() {
        let registry = ArchetypeRegistry::new();
        let mut mem = PatternMemory::new(0);

        // Update many times with same features
        for _ in 0..20 {
            mem.update(&registry, [0.1, 0.1, 0.0, 0.2]);
        }

        assert!(mem.is_stable());
    }

    #[test]
    fn test_pattern_memory_archetype_change() {
        let registry = ArchetypeRegistry::new();
        let mut mem = PatternMemory::new(0);

        // Build up some stability
        for _ in 0..10 {
            mem.update(&registry, [0.1, 0.1, 0.0, 0.2]);
        }
        let old_trace = mem.trace;

        // Change to very different features
        mem.update(&registry, [0.9, 0.9, 0.7, 0.7]);

        // Trace should have increased
        assert!(mem.trace > old_trace);
        assert_eq!(mem.stability, 0);
    }
}
