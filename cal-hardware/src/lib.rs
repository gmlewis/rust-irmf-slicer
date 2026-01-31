use anyhow::{Result, anyhow};
use serialport::SerialPort;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

pub mod projection_window;

/// Message IDs for Thorlabs APT Protocol
#[allow(dead_code)]
mod msg {
    pub const MOT_MOVE_HOME: u16 = 0x0443;
    pub const MOT_MOVE_VELOCITY: u16 = 0x0457;
    pub const MOT_STOP_IMMEDIATE: u16 = 0x0465;
    pub const MOT_REQ_POSN: u16 = 0x0411;
    pub const MOT_GET_POSN: u16 = 0x0412;
    pub const MOT_MOVE_COMPLETED: u16 = 0x0464;
}

/// Trait for a rotation stage motor.
pub trait Motor {
    /// Returns the current position in degrees (0.0 to 360.0).
    fn get_position(&mut self) -> Result<f32>;
    /// Starts rotation at a given velocity (deg/sec).
    fn start_rotation(&mut self, velocity: f32) -> Result<()>;
    /// Stops rotation immediately.
    fn stop(&mut self) -> Result<()>;
}

/// Implementation of a Thorlabs motor stage via Serial (APT Protocol).
pub struct ThorlabsMotor {
    port: Box<dyn SerialPort>,
    counts_per_deg: f32, // Conversion factor (device dependent)
    channel: u8,
}

impl ThorlabsMotor {
    pub fn new(port_name: &str) -> Result<Self> {
        let port = serialport::new(port_name, 115_200)
            .timeout(Duration::from_millis(50))
            .open()?;

        Ok(Self {
            port,
            counts_per_deg: 1919.64, // Default for many Thorlabs PRMT1 stages
            channel: 1,
        })
    }

    /// Sends a short 6-byte message
    fn send_command(&mut self, msg_id: u16, param1: u8, param2: u8) -> Result<()> {
        let mut buf = [0u8; 6];
        buf[0..2].copy_from_slice(&msg_id.to_le_bytes());
        buf[2] = param1;
        buf[3] = param2;
        buf[4] = 0x50; // Destination: Generic USB unit
        buf[5] = 0x01; // Source: Host
        self.port.write_all(&buf)?;
        Ok(())
    }

    /// Receives and parses a GET_POSN message
    fn read_position_response(&mut self) -> Result<f32> {
        let mut header = [0u8; 6];
        self.port.read_exact(&mut header)?;

        let msg_id = u16::from_le_bytes([header[0], header[1]]);
        if msg_id != msg::MOT_GET_POSN {
            return Err(anyhow!("Unexpected message ID: 0x{:04X}", msg_id));
        }

        // MOT_GET_POSN has 6 bytes of data following the header
        let mut data = [0u8; 6];
        self.port.read_exact(&mut data)?;

        // Position is a 32-bit signed integer in the first 4 bytes of data
        let counts = i32::from_le_bytes([data[2], data[3], data[4], data[5]]);

        Ok(counts as f32 / self.counts_per_deg)
    }
}

impl Motor for ThorlabsMotor {
    fn get_position(&mut self) -> Result<f32> {
        self.send_command(msg::MOT_REQ_POSN, self.channel, 0)?;
        self.read_position_response()
    }

    fn start_rotation(&mut self, velocity: f32) -> Result<()> {
        // Simple version: Param1 is channel, Param2 is direction (1 for forward)
        // Note: Real velocity control requires a longer data packet (MOT_SET_VELPARAMS)
        // and then MOT_MOVE_VELOCITY.
        self.send_command(
            msg::MOT_MOVE_VELOCITY,
            self.channel,
            if velocity >= 0.0 { 1 } else { 2 },
        )
    }

    fn stop(&mut self) -> Result<()> {
        self.send_command(msg::MOT_STOP_IMMEDIATE, self.channel, 0)
    }
}

/// A mock motor for testing without hardware.
pub struct MockMotor {
    start_time: Option<std::time::Instant>,
    velocity: f32,
}

impl MockMotor {
    pub fn new() -> Self {
        Self {
            start_time: None,
            velocity: 0.0,
        }
    }
}

impl Default for MockMotor {
    fn default() -> Self {
        Self::new()
    }
}

impl Motor for MockMotor {
    fn get_position(&mut self) -> Result<f32> {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed().as_secs_f32();
            Ok((elapsed * self.velocity) % 360.0)
        } else {
            Ok(0.0)
        }
    }

    fn start_rotation(&mut self, velocity: f32) -> Result<()> {
        self.start_time = Some(std::time::Instant::now());
        self.velocity = velocity;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        self.start_time = None;
        Ok(())
    }
}

/// Orchestrates the synchronization between motor and display.
pub struct ProjectionController<M: Motor + Send + 'static> {
    pub motor: Arc<std::sync::Mutex<M>>,
    _projections: ndarray::Array3<f32>, // (nr, n_angles, nz)
    angles: Vec<f32>,
    shared_frame_index: Arc<AtomicUsize>,
}

impl<M: Motor + Send + 'static> ProjectionController<M> {
    pub fn new(motor: M, projections: ndarray::Array3<f32>, angles: Vec<f32>) -> Self {
        Self {
            motor: Arc::new(std::sync::Mutex::new(motor)),
            _projections: projections,
            angles,
            shared_frame_index: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn shared_frame_index(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.shared_frame_index)
    }

    /// Finds the closest projection frame index for a given motor angle.
    pub fn get_frame_index(&self, current_angle: f32) -> usize {
        let mut closest_idx = 0;
        let mut min_diff = f32::MAX;

        for (i, &angle) in self.angles.iter().enumerate() {
            let diff = (angle - current_angle).abs();
            let wrapped_diff = diff.min(360.0 - diff);
            if wrapped_diff < min_diff {
                min_diff = wrapped_diff;
                closest_idx = i;
            }
        }
        closest_idx
    }

    /// Spawns a background thread to poll the motor and update the shared frame index.
    pub fn spawn_sync_thread(&self, stop_signal: Arc<AtomicBool>) {
        let motor = Arc::clone(&self.motor);
        let shared_idx = Arc::clone(&self.shared_frame_index);
        let angles = self.angles.clone();

        std::thread::spawn(move || {
            while !stop_signal.load(Ordering::Relaxed) {
                if let Ok(mut m) = motor.lock() {
                    if let Ok(pos) = m.get_position() {
                        // Find closest index
                        let mut closest_idx = 0;
                        let mut min_diff = f32::MAX;
                        for (i, &angle) in angles.iter().enumerate() {
                            let diff = (angle - pos).abs();
                            let wrapped_diff = diff.min(360.0 - diff);
                            if wrapped_diff < min_diff {
                                min_diff = wrapped_diff;
                                closest_idx = i;
                            }
                        }
                        shared_idx.store(closest_idx, Ordering::Relaxed);
                    }
                }
                std::thread::sleep(Duration::from_millis(5)); // ~200Hz polling
            }
        });
    }

    pub fn poll_and_render(&mut self, window: &projection_window::ProjectionWindow) -> Result<()> {
        let frame_idx = self.shared_frame_index.load(Ordering::Relaxed);
        window.render(frame_idx)?;
        Ok(())
    }

    pub fn run_sync_loop(&mut self) -> Result<()> {
        {
            let mut m = self
                .motor
                .lock()
                .map_err(|_| anyhow!("Mutex lock failed"))?;
            m.start_rotation(10.0)?; // Example 10 deg/sec
        }

        loop {
            let frame_idx = self.shared_frame_index.load(Ordering::Relaxed);

            // In a real implementation, we would send the frame_idx to the display thread/GPU.
            println!("Frame Index: {}", frame_idx);

            std::thread::sleep(Duration::from_millis(16)); // ~60Hz polling
        }
    }
}
