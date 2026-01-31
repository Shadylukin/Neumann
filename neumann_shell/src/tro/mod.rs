// SPDX-License-Identifier: MIT OR Apache-2.0
//! Tensor Rust Organism (TRO) - Living CLI Border
//!
//! A hybrid cellular automaton that creates organic, living border patterns
//! around the terminal. Combines Physarum (slime mold) simulation with
//! reaction-diffusion for an industrial horror aesthetic.

#![allow(dead_code)]
#![allow(clippy::missing_const_for_fn)]

mod activity;
mod archetype;
mod boot;
mod border_region;
mod charset;
mod effects;
mod organism;
mod palette;
mod physarum;
mod reaction;
mod render;
mod tendril;

pub use activity::{ActivitySensor, OpType};
pub use boot::{BootSequence, BootStyle};
pub use border_region::{BorderCell, BorderCoord, BorderRegion, ResolutionMode};
pub use charset::CharsetMode;
pub use organism::{TroCell, TroState};
pub use palette::Palette;
pub use render::TroRenderer;
pub use tendril::{Tendril, TendrilConfig, TendrilManager};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use parking_lot::RwLock;

/// Commands that can be sent to the TRO render thread.
#[derive(Debug, Clone)]
pub enum TroCommand {
    /// Pause rendering (preserves state).
    Pause,
    /// Resume rendering.
    Resume,
    /// Trigger activity pulse at position.
    Pulse { position: usize, intensity: f32 },
    /// Trigger error glitch effect.
    Glitch { duration_ms: u32 },
    /// Shutdown the render thread.
    Shutdown,
    /// Resize the border dimensions.
    Resize { width: u16, height: u16 },
    /// Change the color palette.
    SetPalette(Palette),
    /// Toggle CRT effects.
    SetCrtEffects(bool),
    /// Set character set mode.
    SetCharsetMode(CharsetMode),
    /// Set resolution mode for sub-pixel rendering.
    SetResolution(ResolutionMode),
    /// Set border depth (number of cell layers).
    SetBorderDepth(u8),
}

/// Configuration for the TRO system.
#[derive(Debug, Clone)]
pub struct TroConfig {
    /// Enable TRO border (auto-disabled for non-TTY).
    pub enabled: bool,
    /// Target frames per second (default: 15).
    pub fps: u32,
    /// Number of Physarum agents (default: 2000).
    pub agent_count: usize,
    /// Color palette to use.
    pub palette: Palette,
    /// Enable CRT effects (scanlines, flicker).
    pub crt_effects: bool,
    /// Boot sequence style.
    pub boot_style: BootStyle,
    /// Border width in characters.
    pub border_width: u16,
    /// Reaction-diffusion update frequency (every N frames).
    pub reaction_frequency: u32,
    /// Character set mode (Unicode or ASCII fallback).
    pub charset_mode: CharsetMode,
    /// Border depth in cells (default: 1, max: 5).
    pub border_depth: u8,
    /// Resolution mode for sub-pixel rendering.
    pub resolution: ResolutionMode,
    /// Tendril configuration.
    pub tendril_config: TendrilConfig,
}

impl Default for TroConfig {
    fn default() -> Self {
        Self {
            enabled: console::Term::stdout().is_term(),
            fps: 15,
            agent_count: 2000,
            palette: Palette::PhosphorGreen,
            crt_effects: true,
            boot_style: BootStyle::Full,
            border_width: 1,
            reaction_frequency: 4,
            charset_mode: CharsetMode::detect(),
            border_depth: 1,
            resolution: ResolutionMode::default(),
            tendril_config: TendrilConfig::default(),
        }
    }
}

impl TroConfig {
    /// Creates a minimal config for low-resource environments.
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            enabled: true,
            fps: 10,
            agent_count: 500,
            palette: Palette::PhosphorGreen,
            crt_effects: false,
            boot_style: BootStyle::Compact,
            border_width: 1,
            reaction_frequency: 8,
            charset_mode: CharsetMode::Ascii, // Minimal uses ASCII for compatibility
            border_depth: 1,
            resolution: ResolutionMode::Standard,
            tendril_config: TendrilConfig::minimal(),
        }
    }

    /// Disables TRO completely.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Controller for the TRO system.
///
/// Manages the render thread and provides the interface for database activity
/// to influence the visual display.
pub struct TroController {
    state: Arc<RwLock<TroState>>,
    activity_sensor: Arc<ActivitySensor>,
    cmd_tx: Sender<TroCommand>,
    running: Arc<AtomicBool>,
    render_thread: Option<JoinHandle<()>>,
    config: TroConfig,
}

impl TroController {
    /// Creates a new TRO controller with the given configuration.
    #[must_use]
    pub fn new(config: TroConfig) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded();

        let term_size = console::Term::stdout().size();
        let width = term_size.1;
        let height = term_size.0;

        let state = Arc::new(RwLock::new(TroState::new(
            width,
            height,
            config.agent_count,
            config.palette,
        )));
        let activity_sensor = Arc::new(ActivitySensor::new(
            width as usize * 2 + height as usize * 2,
        ));
        let running = Arc::new(AtomicBool::new(false));

        // Border animation disabled - it conflicts with terminal line output
        // Boot sequence still runs via run_boot_sequence()
        // Keep render thread infrastructure for potential future TUI mode
        let _ = cmd_rx;

        Self {
            state,
            activity_sensor,
            cmd_tx,
            running,
            render_thread: None,
            config,
        }
    }

    /// Spawns the background render thread.
    /// Currently disabled - kept for potential future TUI mode.
    #[allow(dead_code)]
    fn spawn_render_thread(&mut self, cmd_rx: Receiver<TroCommand>) {
        let state = Arc::clone(&self.state);
        let activity = Arc::clone(&self.activity_sensor);
        let running = Arc::clone(&self.running);
        let fps = self.config.fps;
        let reaction_freq = self.config.reaction_frequency;
        let crt_effects = self.config.crt_effects;

        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            render_loop(
                state,
                activity,
                running,
                cmd_rx,
                fps,
                reaction_freq,
                crt_effects,
            );
        });

        self.render_thread = Some(handle);
    }

    /// Returns a clone of the activity sensor for database hooks.
    #[must_use]
    pub fn activity_sensor(&self) -> Arc<ActivitySensor> {
        Arc::clone(&self.activity_sensor)
    }

    /// Sends a command to the render thread.
    pub fn send(&self, cmd: TroCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// Pauses TRO rendering.
    pub fn pause(&self) {
        self.send(TroCommand::Pause);
    }

    /// Resumes TRO rendering.
    pub fn resume(&self) {
        self.send(TroCommand::Resume);
    }

    /// Triggers a pulse at a specific border position.
    pub fn pulse(&self, position: usize, intensity: f32) {
        self.send(TroCommand::Pulse {
            position,
            intensity,
        });
    }

    /// Triggers an error glitch effect.
    pub fn glitch(&self, duration_ms: u32) {
        self.send(TroCommand::Glitch { duration_ms });
    }

    /// Checks if TRO is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Checks if TRO is currently running.
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Returns the current palette.
    #[must_use]
    pub fn palette(&self) -> Palette {
        self.state.read().palette
    }

    /// Sets the color palette.
    pub fn set_palette(&self, palette: Palette) {
        self.send(TroCommand::SetPalette(palette));
    }

    /// Enables or disables CRT effects.
    pub fn set_crt_effects(&self, enabled: bool) {
        self.send(TroCommand::SetCrtEffects(enabled));
    }

    /// Returns the current charset mode.
    #[must_use]
    pub fn charset_mode(&self) -> CharsetMode {
        self.config.charset_mode
    }

    /// Sets the character set mode.
    pub fn set_charset_mode(&self, mode: CharsetMode) {
        self.send(TroCommand::SetCharsetMode(mode));
    }

    /// Sets the resolution mode for sub-pixel rendering.
    pub fn set_resolution(&self, mode: ResolutionMode) {
        self.send(TroCommand::SetResolution(mode));
    }

    /// Sets the border depth (number of cell layers).
    pub fn set_border_depth(&self, depth: u8) {
        self.send(TroCommand::SetBorderDepth(depth));
    }

    /// Returns current configuration.
    #[must_use]
    pub fn config(&self) -> &TroConfig {
        &self.config
    }

    /// Runs the boot sequence animation.
    pub fn run_boot_sequence(&self, version: &str) {
        if !self.config.enabled {
            return;
        }

        let boot = BootSequence::new(self.config.boot_style, self.config.palette);
        boot.run(version);
    }

    /// Shuts down the TRO system.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.send(TroCommand::Shutdown);

        if let Some(handle) = self.render_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TroController {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Main render loop running in background thread.
/// Currently disabled - kept for potential future TUI mode.
#[allow(dead_code, clippy::needless_pass_by_value)]
fn render_loop(
    state: Arc<RwLock<TroState>>,
    activity: Arc<ActivitySensor>,
    running: Arc<AtomicBool>,
    cmd_rx: Receiver<TroCommand>,
    fps: u32,
    reaction_frequency: u32,
    crt_effects: bool,
) {
    let frame_time = Duration::from_millis(1000 / u64::from(fps));
    let mut renderer = TroRenderer::new(crt_effects);
    let mut paused = false;
    let mut frame_count: u32 = 0;
    let mut glitch_until: Option<Instant> = None;

    while running.load(Ordering::Relaxed) {
        let frame_start = Instant::now();

        // Process commands (non-blocking)
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd {
                TroCommand::Pause => paused = true,
                TroCommand::Resume => paused = false,
                TroCommand::Pulse {
                    position,
                    intensity,
                } => {
                    let mut st = state.write();
                    st.inject_pulse(position, intensity);
                },
                TroCommand::Glitch { duration_ms } => {
                    glitch_until =
                        Some(Instant::now() + Duration::from_millis(u64::from(duration_ms)));
                },
                TroCommand::Shutdown => {
                    running.store(false, Ordering::SeqCst);
                    return;
                },
                TroCommand::Resize { width, height } => {
                    let mut st = state.write();
                    st.resize(width, height);
                },
                TroCommand::SetPalette(palette) => {
                    let mut st = state.write();
                    st.palette = palette;
                },
                TroCommand::SetCrtEffects(enabled) => {
                    renderer.set_crt_effects(enabled);
                },
                TroCommand::SetCharsetMode(mode) => {
                    let mut st = state.write();
                    st.charset_mode = mode;
                },
                TroCommand::SetResolution(mode) => {
                    let mut st = state.write();
                    st.set_resolution(mode);
                },
                TroCommand::SetBorderDepth(depth) => {
                    let mut st = state.write();
                    st.set_border_depth(depth);
                },
            }
        }

        if paused {
            thread::sleep(frame_time);
            continue;
        }

        // Apply activity sensor heat to state
        {
            let heat = activity.drain_heat();
            let mut st = state.write();
            st.apply_activity_heat(&heat);
        }

        // Update TRO state
        {
            let mut st = state.write();

            // Update Physarum agents every frame
            st.physarum.update();

            // Update reaction-diffusion less frequently
            if frame_count % reaction_frequency == 0 {
                st.reaction.step();
            }

            // Compose layers (for 1D cells)
            st.compose_layers();

            // Update 2D border region and tendrils
            st.update_border_region();
        }

        // Check glitch state
        let in_glitch = glitch_until.is_some_and(|t| Instant::now() < t);

        // Render frame using enhanced 2D border region
        {
            let st = state.read();
            let palette_colors = st.palette.get_colors();
            renderer.render_region(
                &st.border_region,
                &st.tendril_manager,
                palette_colors,
                in_glitch,
            );
        }

        frame_count = frame_count.wrapping_add(1);

        // Maintain frame rate
        let elapsed = frame_start.elapsed();
        if let Some(remaining) = frame_time.checked_sub(elapsed) {
            thread::sleep(remaining);
        }
    }

    // Clear border on exit
    renderer.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tro_config_default() {
        let config = TroConfig::default();
        assert_eq!(config.fps, 15);
        assert_eq!(config.agent_count, 2000);
        assert!(config.crt_effects);
    }

    #[test]
    fn test_tro_config_minimal() {
        let config = TroConfig::minimal();
        assert_eq!(config.fps, 10);
        assert_eq!(config.agent_count, 500);
        assert!(!config.crt_effects);
    }

    #[test]
    fn test_tro_config_disabled() {
        let config = TroConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_tro_command_variants() {
        let _pause = TroCommand::Pause;
        let _resume = TroCommand::Resume;
        let _pulse = TroCommand::Pulse {
            position: 0,
            intensity: 1.0,
        };
        let _glitch = TroCommand::Glitch { duration_ms: 100 };
        let _shutdown = TroCommand::Shutdown;
        let _resize = TroCommand::Resize {
            width: 80,
            height: 24,
        };
        let _set_resolution = TroCommand::SetResolution(ResolutionMode::Braille);
        let _set_depth = TroCommand::SetBorderDepth(3);
    }

    #[test]
    fn test_tro_config_new_fields() {
        let config = TroConfig::default();
        assert_eq!(config.border_depth, 1);
        assert_eq!(config.resolution, ResolutionMode::Standard);
    }

    #[test]
    fn test_tro_config_minimal_new_fields() {
        let config = TroConfig::minimal();
        assert_eq!(config.border_depth, 1);
        assert_eq!(config.resolution, ResolutionMode::Standard);
    }
}
