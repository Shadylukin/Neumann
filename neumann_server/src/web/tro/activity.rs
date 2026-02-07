// SPDX-License-Identifier: BSL-1.1 OR Apache-2.0
//! Activity pulse types for TRO WebSocket communication.
//!
//! Defines the message types sent from server to client to trigger
//! visual activity in the TRO border animation.

use serde::{Deserialize, Serialize};

/// Types of database operations that generate visual activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpType {
    /// Read operation (GET, SELECT, etc).
    Get,
    /// Write operation (PUT, INSERT, etc).
    Put,
    /// Delete operation.
    Delete,
    /// Scan/query operation.
    Scan,
    /// Graph traversal.
    GraphTraversal,
    /// Vector similarity search.
    VectorSearch,
    /// Error condition.
    Error,
    /// Heavy/batch operation.
    Batch,
}

impl OpType {
    /// Returns the base intensity for this operation type.
    #[must_use]
    pub const fn base_intensity(&self) -> f32 {
        match self {
            Self::Get => 0.3,
            Self::Put => 0.6,
            Self::Delete => 0.8,
            Self::Scan => 0.4,
            Self::GraphTraversal | Self::VectorSearch => 0.5,
            Self::Error => 1.0,
            Self::Batch => 0.7,
        }
    }

    /// Returns the spread radius for this operation type.
    #[must_use]
    pub const fn spread_radius(&self) -> u32 {
        match self {
            Self::Get => 10,
            Self::Put => 15,
            Self::Delete => 25,
            Self::Scan => 40,
            Self::GraphTraversal => 35,
            Self::VectorSearch => 30,
            Self::Error => 60,
            Self::Batch => 50,
        }
    }

    /// Returns the particle burst count for this operation.
    #[must_use]
    pub const fn particle_count(&self) -> u32 {
        match self {
            Self::Get => 3,
            Self::Put => 5,
            Self::Delete | Self::VectorSearch => 8,
            Self::Scan => 10,
            Self::GraphTraversal => 12,
            Self::Error => 20,
            Self::Batch => 15,
        }
    }

    /// Returns CSS class name for styling.
    #[must_use]
    pub const fn css_class(&self) -> &'static str {
        match self {
            Self::Get => "tro-pulse-get",
            Self::Put => "tro-pulse-put",
            Self::Delete => "tro-pulse-delete",
            Self::Scan => "tro-pulse-scan",
            Self::GraphTraversal => "tro-pulse-graph",
            Self::VectorSearch => "tro-pulse-vector",
            Self::Error => "tro-pulse-error",
            Self::Batch => "tro-pulse-batch",
        }
    }
}

/// An activity pulse event sent to the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityPulse {
    /// Type of operation that triggered this pulse.
    pub op_type: OpType,
    /// Position along the border (0.0 - 1.0 normalized).
    pub position: f32,
    /// Intensity of the pulse (0.0 - 1.0).
    pub intensity: f32,
    /// Spread radius in border units.
    pub radius: u32,
    /// Number of particles to spawn.
    pub particles: u32,
    /// Timestamp (milliseconds since epoch).
    pub timestamp: u64,
}

impl ActivityPulse {
    /// Creates a new activity pulse.
    #[must_use]
    pub fn new(op_type: OpType, position: f32, timestamp: u64) -> Self {
        Self {
            op_type,
            position: position.clamp(0.0, 1.0),
            intensity: op_type.base_intensity(),
            radius: op_type.spread_radius(),
            particles: op_type.particle_count(),
            timestamp,
        }
    }

    /// Creates a pulse with custom intensity.
    #[must_use]
    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity.clamp(0.0, 1.0);
        self
    }

    /// Creates a pulse with custom radius.
    #[must_use]
    pub const fn with_radius(mut self, radius: u32) -> Self {
        self.radius = radius;
        self
    }

    /// Creates a pulse with custom particle count.
    #[must_use]
    pub const fn with_particles(mut self, particles: u32) -> Self {
        self.particles = particles;
        self
    }
}

/// WebSocket messages for TRO communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TroMessage {
    /// Activity pulse from database operation.
    Pulse(ActivityPulse),
    /// Trigger glitch effect.
    Glitch {
        /// Duration in milliseconds.
        duration_ms: u32,
    },
    /// Update palette.
    SetPalette {
        /// Palette name.
        palette: super::Palette,
    },
    /// Toggle CRT effects.
    SetCrtEffects {
        /// Enable/disable.
        enabled: bool,
    },
    /// Sync state with client.
    StateSync(super::TroState),
    /// Pause animation.
    Pause,
    /// Resume animation.
    Resume,
    /// Configure the simulation.
    Configure(super::TroConfig),
}

impl TroMessage {
    /// Creates a pulse message.
    #[must_use]
    pub fn pulse(op_type: OpType, position: f32, timestamp: u64) -> Self {
        Self::Pulse(ActivityPulse::new(op_type, position, timestamp))
    }

    /// Creates a glitch message.
    #[must_use]
    pub const fn glitch(duration_ms: u32) -> Self {
        Self::Glitch { duration_ms }
    }

    /// Creates a set palette message.
    #[must_use]
    pub const fn set_palette(palette: super::Palette) -> Self {
        Self::SetPalette { palette }
    }

    /// Creates a CRT effects toggle message.
    #[must_use]
    pub const fn set_crt_effects(enabled: bool) -> Self {
        Self::SetCrtEffects { enabled }
    }

    /// Serializes to JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_intensity() {
        assert!(OpType::Get.base_intensity() < OpType::Delete.base_intensity());
        assert!((OpType::Error.base_intensity() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_op_type_spread() {
        assert!(OpType::Get.spread_radius() < OpType::Error.spread_radius());
    }

    #[test]
    fn test_op_type_particle_count() {
        assert!(OpType::Get.particle_count() < OpType::Error.particle_count());
    }

    #[test]
    fn test_op_type_css_class() {
        assert_eq!(OpType::Get.css_class(), "tro-pulse-get");
        assert_eq!(OpType::Error.css_class(), "tro-pulse-error");
    }

    #[test]
    fn test_activity_pulse_new() {
        let pulse = ActivityPulse::new(OpType::Put, 0.5, 12345);
        assert_eq!(pulse.op_type, OpType::Put);
        assert!((pulse.position - 0.5).abs() < f32::EPSILON);
        assert!((pulse.intensity - OpType::Put.base_intensity()).abs() < f32::EPSILON);
        assert_eq!(pulse.radius, OpType::Put.spread_radius());
        assert_eq!(pulse.timestamp, 12345);
    }

    #[test]
    fn test_activity_pulse_position_clamped() {
        let pulse = ActivityPulse::new(OpType::Get, 1.5, 0);
        assert!((pulse.position - 1.0).abs() < f32::EPSILON);

        let pulse2 = ActivityPulse::new(OpType::Get, -0.5, 0);
        assert!(pulse2.position.abs() < f32::EPSILON);
    }

    #[test]
    fn test_activity_pulse_with_methods() {
        let pulse = ActivityPulse::new(OpType::Get, 0.5, 0)
            .with_intensity(0.9)
            .with_radius(50)
            .with_particles(10);

        assert!((pulse.intensity - 0.9).abs() < f32::EPSILON);
        assert_eq!(pulse.radius, 50);
        assert_eq!(pulse.particles, 10);
    }

    #[test]
    fn test_tro_message_pulse() {
        let msg = TroMessage::pulse(OpType::Put, 0.3, 12345);
        if let TroMessage::Pulse(pulse) = msg {
            assert_eq!(pulse.op_type, OpType::Put);
        } else {
            panic!("Expected Pulse variant");
        }
    }

    #[test]
    fn test_tro_message_glitch() {
        let msg = TroMessage::glitch(500);
        if let TroMessage::Glitch { duration_ms } = msg {
            assert_eq!(duration_ms, 500);
        } else {
            panic!("Expected Glitch variant");
        }
    }

    #[test]
    fn test_tro_message_serialization() {
        let msg = TroMessage::pulse(OpType::Put, 0.5, 12345);
        let json = msg.to_json();
        assert!(json.contains("pulse"));
        assert!(json.contains("put"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_tro_message_glitch_serialization() {
        let msg = TroMessage::glitch(100);
        let json = msg.to_json();
        assert!(json.contains("glitch"));
        assert!(json.contains("100"));
    }

    #[test]
    fn test_tro_message_palette_serialization() {
        let msg = TroMessage::set_palette(super::super::Palette::Amber);
        let json = msg.to_json();
        assert!(json.contains("set_palette"));
        assert!(json.contains("amber"));
    }

    #[test]
    fn test_tro_message_crt_serialization() {
        let msg = TroMessage::set_crt_effects(true);
        let json = msg.to_json();
        assert!(json.contains("set_crt_effects"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_op_type_serialization() {
        let op = OpType::GraphTraversal;
        let json = serde_json::to_string(&op).expect("serialization failed");
        assert!(json.contains("graph_traversal"));

        let decoded: OpType = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded, OpType::GraphTraversal);
    }

    #[test]
    fn test_activity_pulse_serialization() {
        let pulse = ActivityPulse::new(OpType::VectorSearch, 0.75, 999);
        let json = serde_json::to_string(&pulse).expect("serialization failed");
        assert!(json.contains("vector_search"));
        assert!(json.contains("0.75"));

        let decoded: ActivityPulse = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(decoded.op_type, OpType::VectorSearch);
        assert!((decoded.position - 0.75).abs() < f32::EPSILON);
    }

    // ========== Additional OpType tests ==========

    #[test]
    fn test_op_type_all_intensities() {
        assert!((OpType::Get.base_intensity() - 0.3).abs() < f32::EPSILON);
        assert!((OpType::Put.base_intensity() - 0.6).abs() < f32::EPSILON);
        assert!((OpType::Delete.base_intensity() - 0.8).abs() < f32::EPSILON);
        assert!((OpType::Scan.base_intensity() - 0.4).abs() < f32::EPSILON);
        assert!((OpType::GraphTraversal.base_intensity() - 0.5).abs() < f32::EPSILON);
        assert!((OpType::VectorSearch.base_intensity() - 0.5).abs() < f32::EPSILON);
        assert!((OpType::Batch.base_intensity() - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_op_type_all_spread_radii() {
        assert_eq!(OpType::Get.spread_radius(), 10);
        assert_eq!(OpType::Put.spread_radius(), 15);
        assert_eq!(OpType::Delete.spread_radius(), 25);
        assert_eq!(OpType::Scan.spread_radius(), 40);
        assert_eq!(OpType::GraphTraversal.spread_radius(), 35);
        assert_eq!(OpType::VectorSearch.spread_radius(), 30);
        assert_eq!(OpType::Error.spread_radius(), 60);
        assert_eq!(OpType::Batch.spread_radius(), 50);
    }

    #[test]
    fn test_op_type_all_particle_counts() {
        assert_eq!(OpType::Get.particle_count(), 3);
        assert_eq!(OpType::Put.particle_count(), 5);
        assert_eq!(OpType::Delete.particle_count(), 8);
        assert_eq!(OpType::Scan.particle_count(), 10);
        assert_eq!(OpType::GraphTraversal.particle_count(), 12);
        assert_eq!(OpType::VectorSearch.particle_count(), 8);
        assert_eq!(OpType::Error.particle_count(), 20);
        assert_eq!(OpType::Batch.particle_count(), 15);
    }

    #[test]
    fn test_op_type_all_css_classes() {
        assert_eq!(OpType::Put.css_class(), "tro-pulse-put");
        assert_eq!(OpType::Delete.css_class(), "tro-pulse-delete");
        assert_eq!(OpType::Scan.css_class(), "tro-pulse-scan");
        assert_eq!(OpType::GraphTraversal.css_class(), "tro-pulse-graph");
        assert_eq!(OpType::VectorSearch.css_class(), "tro-pulse-vector");
        assert_eq!(OpType::Batch.css_class(), "tro-pulse-batch");
    }

    #[test]
    fn test_op_type_all_serialization() {
        for op in [
            OpType::Get,
            OpType::Put,
            OpType::Delete,
            OpType::Scan,
            OpType::GraphTraversal,
            OpType::VectorSearch,
            OpType::Error,
            OpType::Batch,
        ] {
            let json = serde_json::to_string(&op).expect("serialization failed");
            let decoded: OpType = serde_json::from_str(&json).expect("deserialization failed");
            assert_eq!(decoded, op);
        }
    }

    #[test]
    fn test_op_type_debug() {
        let op = OpType::GraphTraversal;
        let debug_str = format!("{:?}", op);
        assert!(debug_str.contains("GraphTraversal"));
    }

    #[test]
    fn test_op_type_clone() {
        let op = OpType::VectorSearch;
        let cloned = op;
        assert_eq!(cloned, op);
    }

    // ========== Additional ActivityPulse tests ==========

    #[test]
    fn test_activity_pulse_all_op_types() {
        for op_type in [
            OpType::Get,
            OpType::Put,
            OpType::Delete,
            OpType::Scan,
            OpType::GraphTraversal,
            OpType::VectorSearch,
            OpType::Error,
            OpType::Batch,
        ] {
            let pulse = ActivityPulse::new(op_type, 0.5, 1000);
            assert_eq!(pulse.op_type, op_type);
            assert!((pulse.intensity - op_type.base_intensity()).abs() < f32::EPSILON);
            assert_eq!(pulse.radius, op_type.spread_radius());
            assert_eq!(pulse.particles, op_type.particle_count());
        }
    }

    #[test]
    fn test_activity_pulse_with_intensity_clamped() {
        let pulse = ActivityPulse::new(OpType::Get, 0.5, 0).with_intensity(1.5);
        assert!((pulse.intensity - 1.0).abs() < f32::EPSILON);

        let pulse2 = ActivityPulse::new(OpType::Get, 0.5, 0).with_intensity(-0.5);
        assert!(pulse2.intensity.abs() < f32::EPSILON);
    }

    #[test]
    fn test_activity_pulse_clone() {
        let pulse = ActivityPulse::new(OpType::Put, 0.3, 12345);
        let cloned = pulse.clone();
        assert_eq!(cloned.op_type, OpType::Put);
        assert!((cloned.position - 0.3).abs() < f32::EPSILON);
        assert_eq!(cloned.timestamp, 12345);
    }

    #[test]
    fn test_activity_pulse_debug() {
        let pulse = ActivityPulse::new(OpType::Error, 0.9, 0);
        let debug_str = format!("{:?}", pulse);
        assert!(debug_str.contains("ActivityPulse"));
        assert!(debug_str.contains("Error"));
    }

    // ========== Additional TroMessage tests ==========

    #[test]
    fn test_tro_message_state_sync() {
        let state = super::super::TroState::new(10);
        let msg = TroMessage::StateSync(state);
        let json = msg.to_json();
        assert!(json.contains("state_sync"));
        assert!(json.contains("pheromone"));
    }

    #[test]
    fn test_tro_message_pause() {
        let msg = TroMessage::Pause;
        let json = msg.to_json();
        assert!(json.contains("pause"));
    }

    #[test]
    fn test_tro_message_resume() {
        let msg = TroMessage::Resume;
        let json = msg.to_json();
        assert!(json.contains("resume"));
    }

    #[test]
    fn test_tro_message_configure() {
        let config = super::super::TroConfig::default();
        let msg = TroMessage::Configure(config);
        let json = msg.to_json();
        assert!(json.contains("configure"));
        assert!(json.contains("enabled"));
    }

    #[test]
    fn test_tro_message_clone() {
        let msg = TroMessage::glitch(100);
        let cloned = msg.clone();
        if let TroMessage::Glitch { duration_ms } = cloned {
            assert_eq!(duration_ms, 100);
        } else {
            panic!("Expected Glitch variant");
        }
    }

    #[test]
    fn test_tro_message_debug() {
        let msg = TroMessage::Pause;
        let debug_str = format!("{:?}", msg);
        assert!(debug_str.contains("Pause"));
    }

    #[test]
    fn test_tro_message_all_variants_serialization() {
        // Test each variant can be serialized
        let messages = vec![
            TroMessage::pulse(OpType::Get, 0.5, 1000),
            TroMessage::glitch(200),
            TroMessage::set_palette(super::super::Palette::Rust),
            TroMessage::set_crt_effects(false),
            TroMessage::Pause,
            TroMessage::Resume,
        ];

        for msg in messages {
            let json = msg.to_json();
            assert!(!json.is_empty());
            assert!(json.starts_with('{'));
        }
    }
}
