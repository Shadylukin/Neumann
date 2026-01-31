// SPDX-License-Identifier: MIT OR Apache-2.0
//! Activity sensor for coupling database operations to TRO visuals.
//!
//! Tracks database activity and converts it to visual heat that
//! influences the TRO border animation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Types of database operations that generate visual activity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    pub const fn spread_radius(&self) -> usize {
        match self {
            Self::Get => 2,
            Self::Put => 3,
            Self::Delete => 5,
            Self::Scan => 10,
            Self::GraphTraversal => 8,
            Self::VectorSearch => 6,
            Self::Error => 15,
            Self::Batch => 12,
        }
    }
}

/// Sensor for tracking database activity and mapping to border positions.
pub struct ActivitySensor {
    /// Heat map around the border (atomic for lock-free access).
    heat_map: Vec<AtomicU32>,
    /// Operation counters by type.
    op_counts: [AtomicU64; 8],
    /// Total operations since last reset.
    total_ops: AtomicU64,
    /// Border length.
    border_length: usize,
}

impl ActivitySensor {
    /// Creates a new activity sensor for the given border length.
    #[must_use]
    pub fn new(border_length: usize) -> Self {
        let heat_map = (0..border_length).map(|_| AtomicU32::new(0)).collect();

        Self {
            heat_map,
            op_counts: [
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
                AtomicU64::new(0),
            ],
            total_ops: AtomicU64::new(0),
            border_length,
        }
    }

    /// Records a database operation.
    ///
    /// This is called from `QueryRouter` on every operation to generate
    /// visual activity in the TRO border.
    pub fn record(&self, op: OpType, key: &str) {
        // Update operation counter
        let op_idx = match op {
            OpType::Get => 0,
            OpType::Put => 1,
            OpType::Delete => 2,
            OpType::Scan => 3,
            OpType::GraphTraversal => 4,
            OpType::VectorSearch => 5,
            OpType::Error => 6,
            OpType::Batch => 7,
        };
        self.op_counts[op_idx].fetch_add(1, Ordering::Relaxed);
        self.total_ops.fetch_add(1, Ordering::Relaxed);

        if self.border_length == 0 {
            return;
        }

        // Hash key to border position
        let position = self.hash_to_position(key);
        let intensity = op.base_intensity();
        let radius = op.spread_radius();

        // Apply heat with spread
        self.apply_heat(position, intensity, radius);
    }

    /// Records an error event (causes glitch effect).
    pub fn record_error(&self, message: &str) {
        self.record(OpType::Error, message);
    }

    /// Records a batch operation start.
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for visual heat calculation
    pub fn record_batch_start(&self, count: usize) {
        // Distribute heat around the border proportional to batch size
        let intensity = (count as f32 / 100.0).min(1.0);
        for i in 0..self.border_length {
            if i % 10 == 0 {
                self.apply_heat(i, intensity * 0.3, 3);
            }
        }
        self.record(OpType::Batch, &count.to_string());
    }

    /// Hashes a key to a border position.
    #[allow(clippy::cast_possible_truncation)] // Truncation is intentional for hash distribution
    fn hash_to_position(&self, key: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.border_length
    }

    /// Applies heat at a position with spread to neighbors.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn apply_heat(&self, center: usize, intensity: f32, radius: usize) {
        // intensity is always 0.0-1.0, so heat_value is always 0-1000
        let heat_value = (intensity * 1000.0) as u32;

        for offset in 0..=radius {
            let decay = 1.0 - (offset as f32 / (radius + 1) as f32);
            let value = (heat_value as f32 * decay) as u32;

            if value == 0 {
                continue;
            }

            // Apply to both sides
            let left = (center + self.border_length - offset) % self.border_length;
            let right = (center + offset) % self.border_length;

            self.heat_map[left].fetch_add(value, Ordering::Relaxed);
            if offset > 0 {
                self.heat_map[right].fetch_add(value, Ordering::Relaxed);
            }
        }
    }

    /// Drains the heat map and returns normalized values.
    ///
    /// This is called by the render loop to get heat values and reset them.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for visual heat values
    pub fn drain_heat(&self) -> Vec<f32> {
        self.heat_map
            .iter()
            .map(|h| {
                let raw = h.swap(0, Ordering::Relaxed);
                (raw as f32 / 1000.0).min(1.0)
            })
            .collect()
    }

    /// Gets the current heat at a position without draining.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for visual heat values
    pub fn peek_heat(&self, position: usize) -> f32 {
        self.heat_map.get(position).map_or(0.0, |h| {
            (h.load(Ordering::Relaxed) as f32 / 1000.0).min(1.0)
        })
    }

    /// Returns total operations recorded.
    #[must_use]
    pub fn total_ops(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }

    /// Returns operation count by type.
    #[must_use]
    pub fn op_count(&self, op: OpType) -> u64 {
        let idx = match op {
            OpType::Get => 0,
            OpType::Put => 1,
            OpType::Delete => 2,
            OpType::Scan => 3,
            OpType::GraphTraversal => 4,
            OpType::VectorSearch => 5,
            OpType::Error => 6,
            OpType::Batch => 7,
        };
        self.op_counts[idx].load(Ordering::Relaxed)
    }

    /// Resets all counters and heat.
    pub fn reset(&self) {
        for h in &self.heat_map {
            h.store(0, Ordering::Relaxed);
        }
        for c in &self.op_counts {
            c.store(0, Ordering::Relaxed);
        }
        self.total_ops.store(0, Ordering::Relaxed);
    }

    /// Returns the border length.
    #[must_use]
    pub const fn border_length(&self) -> usize {
        self.border_length
    }

    /// Resizes the heat map to a new border length.
    pub fn resize(&mut self, new_length: usize) {
        if new_length != self.border_length {
            self.heat_map = (0..new_length).map(|_| AtomicU32::new(0)).collect();
            self.border_length = new_length;
        }
    }
}

/// Statistics about activity patterns.
#[derive(Debug, Clone, Default)]
pub struct ActivityStats {
    /// Total operations.
    pub total: u64,
    /// Read operations.
    pub reads: u64,
    /// Write operations.
    pub writes: u64,
    /// Delete operations.
    pub deletes: u64,
    /// Errors.
    pub errors: u64,
    /// Average heat level.
    pub avg_heat: f32,
    /// Peak heat level.
    pub peak_heat: f32,
}

impl ActivityStats {
    /// Computes stats from an activity sensor.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for visual heat stats
    pub fn from_sensor(sensor: &ActivitySensor) -> Self {
        let heat: Vec<f32> = sensor
            .heat_map
            .iter()
            .map(|h| (h.load(Ordering::Relaxed) as f32 / 1000.0).min(1.0))
            .collect();

        let avg_heat = if heat.is_empty() {
            0.0
        } else {
            heat.iter().sum::<f32>() / heat.len() as f32
        };

        let peak_heat = heat
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        Self {
            total: sensor.total_ops(),
            reads: sensor.op_count(OpType::Get),
            writes: sensor.op_count(OpType::Put),
            deletes: sensor.op_count(OpType::Delete),
            errors: sensor.op_count(OpType::Error),
            avg_heat,
            peak_heat,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_type_intensity() {
        assert!(OpType::Get.base_intensity() < OpType::Delete.base_intensity());
        assert_eq!(OpType::Error.base_intensity(), 1.0);
    }

    #[test]
    fn test_op_type_spread() {
        assert!(OpType::Get.spread_radius() < OpType::Error.spread_radius());
    }

    #[test]
    fn test_sensor_new() {
        let sensor = ActivitySensor::new(100);
        assert_eq!(sensor.border_length(), 100);
        assert_eq!(sensor.total_ops(), 0);
    }

    #[test]
    fn test_sensor_record() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Get, "test_key");
        assert_eq!(sensor.total_ops(), 1);
        assert_eq!(sensor.op_count(OpType::Get), 1);
    }

    #[test]
    fn test_sensor_record_multiple() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Get, "key1");
        sensor.record(OpType::Put, "key2");
        sensor.record(OpType::Get, "key3");

        assert_eq!(sensor.total_ops(), 3);
        assert_eq!(sensor.op_count(OpType::Get), 2);
        assert_eq!(sensor.op_count(OpType::Put), 1);
    }

    #[test]
    fn test_sensor_heat_spread() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Put, "test");

        // Should have heat at multiple positions due to spread
        let heat = sensor.drain_heat();
        let non_zero: usize = heat.iter().filter(|&&h| h > 0.0).count();
        assert!(non_zero > 1);
    }

    #[test]
    fn test_sensor_drain_heat() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Put, "test");

        let heat1 = sensor.drain_heat();
        let total1: f32 = heat1.iter().sum();
        assert!(total1 > 0.0);

        // After drain, heat should be cleared
        let heat2 = sensor.drain_heat();
        let total2: f32 = heat2.iter().sum();
        assert_eq!(total2, 0.0);
    }

    #[test]
    fn test_sensor_peek_heat() {
        let sensor = ActivitySensor::new(100);
        sensor.apply_heat(50, 0.5, 3);

        let heat = sensor.peek_heat(50);
        assert!(heat > 0.0);

        // Peek doesn't drain
        let heat2 = sensor.peek_heat(50);
        assert_eq!(heat, heat2);
    }

    #[test]
    fn test_sensor_reset() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Get, "test");
        sensor.record(OpType::Put, "test2");

        assert!(sensor.total_ops() > 0);
        sensor.reset();
        assert_eq!(sensor.total_ops(), 0);
        assert_eq!(sensor.op_count(OpType::Get), 0);
    }

    #[test]
    fn test_sensor_empty_border() {
        let sensor = ActivitySensor::new(0);
        sensor.record(OpType::Get, "test"); // Should not panic
        assert_eq!(sensor.total_ops(), 1);
    }

    #[test]
    fn test_sensor_record_error() {
        let sensor = ActivitySensor::new(100);
        sensor.record_error("test error");
        assert_eq!(sensor.op_count(OpType::Error), 1);
    }

    #[test]
    fn test_sensor_record_batch() {
        let sensor = ActivitySensor::new(100);
        sensor.record_batch_start(50);
        assert_eq!(sensor.op_count(OpType::Batch), 1);
    }

    #[test]
    fn test_sensor_hash_deterministic() {
        let sensor = ActivitySensor::new(100);
        let pos1 = sensor.hash_to_position("test_key");
        let pos2 = sensor.hash_to_position("test_key");
        assert_eq!(pos1, pos2);
    }

    #[test]
    fn test_sensor_hash_distribution() {
        let sensor = ActivitySensor::new(100);
        let positions: Vec<usize> = (0..100)
            .map(|i| sensor.hash_to_position(&format!("key_{i}")))
            .collect();

        // Should have some variety in positions
        let unique: std::collections::HashSet<_> = positions.iter().collect();
        assert!(unique.len() > 10);
    }

    #[test]
    fn test_activity_stats() {
        let sensor = ActivitySensor::new(100);
        sensor.record(OpType::Get, "k1");
        sensor.record(OpType::Put, "k2");
        sensor.record(OpType::Error, "err");

        let stats = ActivityStats::from_sensor(&sensor);
        assert_eq!(stats.total, 3);
        assert_eq!(stats.reads, 1);
        assert_eq!(stats.writes, 1);
        assert_eq!(stats.errors, 1);
    }
}
