// SPDX-License-Identifier: MIT OR Apache-2.0
//! Physarum (slime mold) agent-based simulation.
//!
//! Simulates the foraging behavior of Physarum polycephalum slime mold.
//! Agents move through the border, depositing pheromone trails that
//! influence other agents, creating organic network patterns.

// Physarum simulation involves many float/int conversions for positions.
// These are all intentional for the simulation. Floating point math is kept
// in readable form rather than using mul_add everywhere.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops
)]

use rand::Rng;

/// Configuration for Physarum simulation.
#[derive(Debug, Clone)]
pub struct PhysarumConfig {
    /// Angle offset for sensor positions (radians).
    pub sensor_angle: f32,
    /// Distance ahead to sense pheromone.
    pub sensor_distance: f32,
    /// Maximum rotation per step (radians).
    pub rotation_angle: f32,
    /// Movement speed (cells per step).
    pub speed: f32,
    /// Amount of pheromone deposited per step.
    pub deposit_amount: f32,
    /// Pheromone decay rate per step (0.0-1.0).
    pub decay_rate: f32,
    /// Diffusion rate (spread to neighbors).
    pub diffusion_rate: f32,
}

impl Default for PhysarumConfig {
    fn default() -> Self {
        Self {
            sensor_angle: std::f32::consts::PI / 4.0, // 45 degrees
            sensor_distance: 3.0,
            rotation_angle: std::f32::consts::PI / 6.0, // 30 degrees
            speed: 1.0,
            deposit_amount: 0.8,
            decay_rate: 0.98,
            diffusion_rate: 0.1,
        }
    }
}

/// A single Physarum agent (virtual slime mold particle).
#[derive(Debug, Clone)]
pub struct PhysarumAgent {
    /// Position along the border (0.0 to `border_length`).
    pub position: f32,
    /// Direction of movement (-1.0 or 1.0, since border is 1D).
    pub direction: f32,
    /// Current speed multiplier.
    pub speed: f32,
    /// Whether agent is active.
    pub active: bool,
}

impl PhysarumAgent {
    /// Creates a new agent at a random position.
    #[must_use]
    pub fn new_random(border_length: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = if border_length > 0 {
            rng.gen::<f32>() * border_length as f32
        } else {
            0.0
        };
        Self {
            position,
            direction: if rng.gen_bool(0.5) { 1.0 } else { -1.0 },
            speed: 0.5 + rng.gen::<f32>() * 0.5,
            active: true,
        }
    }

    /// Creates a new agent at a specific position.
    #[must_use]
    pub fn at_position(position: f32, direction: f32) -> Self {
        Self {
            position,
            direction,
            speed: 1.0,
            active: true,
        }
    }
}

/// The Physarum pheromone field and agent collection.
pub struct PhysarumField {
    /// Length of the border.
    border_length: usize,
    /// Pheromone concentration at each cell.
    pheromone: Vec<f32>,
    /// Temporary buffer for diffusion calculation.
    pheromone_buffer: Vec<f32>,
    /// Active agents.
    agents: Vec<PhysarumAgent>,
    /// Configuration.
    config: PhysarumConfig,
}

impl PhysarumField {
    /// Creates a new Physarum field.
    #[must_use]
    pub fn new(border_length: usize, agent_count: usize) -> Self {
        let pheromone = vec![0.0; border_length];
        let pheromone_buffer = vec![0.0; border_length];

        let agents = (0..agent_count)
            .map(|_| PhysarumAgent::new_random(border_length))
            .collect();

        Self {
            border_length,
            pheromone,
            pheromone_buffer,
            agents,
            config: PhysarumConfig::default(),
        }
    }

    /// Creates a field with custom configuration.
    #[must_use]
    pub fn with_config(border_length: usize, agent_count: usize, config: PhysarumConfig) -> Self {
        let mut field = Self::new(border_length, agent_count);
        field.config = config;
        field
    }

    /// Updates all agents and the pheromone field.
    pub fn update(&mut self) {
        if self.border_length == 0 {
            return;
        }

        // Phase 1: Move agents and deposit pheromone
        self.update_agents();

        // Phase 2: Diffuse and decay pheromone
        self.diffuse_and_decay();
    }

    /// Updates all agent positions based on pheromone sensing.
    fn update_agents(&mut self) {
        let border_length = self.border_length;
        let border_length_f32 = border_length as f32;
        let sensor_distance = self.config.sensor_distance;
        let config_speed = self.config.speed;
        let deposit_amount = self.config.deposit_amount;

        // Pre-compute pheromone sampling closure
        let wrap_position = |pos: f32| -> f32 {
            if border_length_f32 <= 0.0 {
                return 0.0;
            }
            ((pos % border_length_f32) + border_length_f32) % border_length_f32
        };

        // Collect positions for deposition
        let mut deposits: Vec<(usize, f32)> = Vec::with_capacity(self.agents.len());

        for agent in &mut self.agents {
            if !agent.active {
                continue;
            }

            // Sense pheromone ahead in three directions
            let left_pos = wrap_position(agent.position + agent.direction * sensor_distance * 0.7);
            let center_pos = wrap_position(agent.position + agent.direction * sensor_distance);
            let right_pos = wrap_position(agent.position + agent.direction * sensor_distance * 1.3);

            let left = self
                .pheromone
                .get(left_pos as usize % border_length)
                .copied()
                .unwrap_or(0.0);
            let center = self
                .pheromone
                .get(center_pos as usize % border_length)
                .copied()
                .unwrap_or(0.0);
            let right = self
                .pheromone
                .get(right_pos as usize % border_length)
                .copied()
                .unwrap_or(0.0);

            // Decide direction based on pheromone gradient
            if center >= left && center >= right {
                // Keep going straight (maybe small random perturbation)
                if rand::thread_rng().gen::<f32>() < 0.1 {
                    agent.direction *= if rand::thread_rng().gen_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    };
                }
            } else if left > right {
                // Stronger pheromone to the "left" (backwards)
                agent.direction = -agent.direction.abs();
            } else if right > left {
                // Stronger pheromone to the "right" (forwards)
                agent.direction = agent.direction.abs();
            } else {
                // Equal - random choice
                agent.direction = if rand::thread_rng().gen_bool(0.5) {
                    1.0
                } else {
                    -1.0
                };
            }

            // Move forward
            let movement = agent.direction * agent.speed * config_speed;
            agent.position = wrap_position(agent.position + movement);

            // Record deposit position
            if border_length > 0 {
                let idx = agent.position as usize % border_length;
                deposits.push((idx, deposit_amount));
            }
        }

        // Apply pheromone deposits
        for (idx, amount) in deposits {
            self.pheromone[idx] = (self.pheromone[idx] + amount).min(1.0);
        }
    }

    /// Diffuses pheromone to neighbors and applies decay.
    fn diffuse_and_decay(&mut self) {
        let len = self.border_length;
        if len == 0 {
            return;
        }

        // Copy current state to buffer
        self.pheromone_buffer.copy_from_slice(&self.pheromone);

        // Apply diffusion using a 3-cell kernel
        for i in 0..len {
            let prev = if i == 0 { len - 1 } else { i - 1 };
            let next = (i + 1) % len;

            let diffused = self.pheromone_buffer[i] * (1.0 - self.config.diffusion_rate)
                + (self.pheromone_buffer[prev] + self.pheromone_buffer[next])
                    * (self.config.diffusion_rate / 2.0);

            // Apply decay
            self.pheromone[i] = diffused * self.config.decay_rate;
        }
    }

    /// Wraps a position to stay within bounds.
    fn wrap_position(&self, pos: f32) -> f32 {
        let len = self.border_length as f32;
        if len <= 0.0 {
            return 0.0;
        }
        ((pos % len) + len) % len
    }

    /// Samples pheromone at a (possibly fractional) position.
    fn sample_pheromone(&self, pos: f32) -> f32 {
        if self.border_length == 0 {
            return 0.0;
        }

        let idx = pos as usize % self.border_length;
        self.pheromone[idx]
    }

    /// Gets the pheromone level at a specific cell.
    #[must_use]
    pub fn get_pheromone(&self, index: usize) -> f32 {
        self.pheromone.get(index).copied().unwrap_or(0.0)
    }

    /// Spawns new agents at a specific position (for activity pulses).
    pub fn spawn_agents_at(&mut self, position: usize, count: usize, intensity: f32) {
        let mut rng = rand::thread_rng();
        for _ in 0..count {
            let offset: f32 = rng.gen::<f32>() * 4.0 - 2.0;
            let pos = self.wrap_position(position as f32 + offset);
            let dir = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
            let mut agent = PhysarumAgent::at_position(pos, dir);
            agent.speed = 0.5 + intensity * 0.5;
            self.agents.push(agent);
        }

        // Keep agent count bounded
        let max_agents = self.border_length * 20;
        if self.agents.len() > max_agents {
            self.agents.drain(0..(self.agents.len() - max_agents));
        }
    }

    /// Resizes the field to a new border length.
    pub fn resize(&mut self, new_length: usize) {
        if new_length == self.border_length {
            return;
        }

        self.pheromone.resize(new_length, 0.0);
        self.pheromone_buffer.resize(new_length, 0.0);
        self.border_length = new_length;

        // Rescale agent positions
        if self.border_length > 0 {
            for agent in &mut self.agents {
                agent.position %= new_length as f32;
            }
        }
    }

    /// Returns the number of active agents.
    #[must_use]
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }

    /// Clears all pheromone and resets agents.
    pub fn reset(&mut self) {
        self.pheromone.fill(0.0);
        self.pheromone_buffer.fill(0.0);

        for agent in &mut self.agents {
            *agent = PhysarumAgent::new_random(self.border_length);
        }
    }

    /// Returns the maximum pheromone concentration.
    #[must_use]
    pub fn max_pheromone(&self) -> f32 {
        self.pheromone
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    /// Returns the average pheromone concentration.
    #[must_use]
    pub fn avg_pheromone(&self) -> f32 {
        if self.pheromone.is_empty() {
            return 0.0;
        }
        self.pheromone.iter().sum::<f32>() / self.pheromone.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = PhysarumConfig::default();
        assert!(config.sensor_angle > 0.0);
        assert!(config.decay_rate > 0.0);
        assert!(config.decay_rate <= 1.0);
    }

    #[test]
    fn test_agent_new_random() {
        let agent = PhysarumAgent::new_random(100);
        assert!(agent.position >= 0.0);
        assert!(agent.position < 100.0);
        assert!(agent.direction.abs() == 1.0);
        assert!(agent.active);
    }

    #[test]
    fn test_agent_at_position() {
        let agent = PhysarumAgent::at_position(50.0, 1.0);
        assert_eq!(agent.position, 50.0);
        assert_eq!(agent.direction, 1.0);
    }

    #[test]
    fn test_field_new() {
        let field = PhysarumField::new(100, 50);
        assert_eq!(field.border_length, 100);
        assert_eq!(field.agents.len(), 50);
        assert_eq!(field.pheromone.len(), 100);
    }

    #[test]
    fn test_field_update_no_panic() {
        let mut field = PhysarumField::new(100, 50);
        for _ in 0..10 {
            field.update();
        }
    }

    #[test]
    fn test_field_update_empty() {
        let mut field = PhysarumField::new(0, 0);
        field.update(); // Should not panic
    }

    #[test]
    fn test_get_pheromone() {
        let field = PhysarumField::new(100, 0);
        assert_eq!(field.get_pheromone(50), 0.0);
        assert_eq!(field.get_pheromone(1000), 0.0); // Out of bounds returns 0
    }

    #[test]
    fn test_spawn_agents_at() {
        let mut field = PhysarumField::new(100, 10);
        let initial = field.agent_count();
        field.spawn_agents_at(50, 5, 0.5);
        assert_eq!(field.agent_count(), initial + 5);
    }

    #[test]
    fn test_resize() {
        let mut field = PhysarumField::new(100, 50);
        field.resize(200);
        assert_eq!(field.border_length, 200);
        assert_eq!(field.pheromone.len(), 200);
    }

    #[test]
    fn test_resize_same_size() {
        let mut field = PhysarumField::new(100, 50);
        field.resize(100);
        assert_eq!(field.border_length, 100);
    }

    #[test]
    fn test_wrap_position() {
        let field = PhysarumField::new(100, 0);
        assert!((field.wrap_position(50.0) - 50.0).abs() < 0.001);
        assert!((field.wrap_position(150.0) - 50.0).abs() < 0.001);
        assert!((field.wrap_position(-10.0) - 90.0).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let mut field = PhysarumField::new(100, 50);
        // Run a few updates to build up pheromone
        for _ in 0..10 {
            field.update();
        }
        assert!(field.max_pheromone() > 0.0);

        field.reset();
        assert_eq!(field.max_pheromone(), 0.0);
    }

    #[test]
    fn test_max_pheromone() {
        let field = PhysarumField::new(100, 0);
        assert_eq!(field.max_pheromone(), 0.0);
    }

    #[test]
    fn test_avg_pheromone() {
        let field = PhysarumField::new(100, 0);
        assert_eq!(field.avg_pheromone(), 0.0);

        let empty = PhysarumField::new(0, 0);
        assert_eq!(empty.avg_pheromone(), 0.0);
    }

    #[test]
    fn test_pheromone_deposits() {
        let mut field = PhysarumField::new(100, 100);
        // Run updates - agents should deposit pheromone
        for _ in 0..50 {
            field.update();
        }
        // After 50 updates, there should be some pheromone buildup
        assert!(field.max_pheromone() > 0.0);
    }

    #[test]
    fn test_pheromone_decays() {
        let mut field = PhysarumField::new(100, 0);
        // Manually set some pheromone
        field.pheromone[50] = 1.0;

        // Update multiple times (no agents to deposit)
        for _ in 0..100 {
            field.diffuse_and_decay();
        }

        // Pheromone should have decayed significantly
        assert!(field.get_pheromone(50) < 0.1);
    }
}
