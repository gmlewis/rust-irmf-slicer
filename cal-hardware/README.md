# cal-hardware

This crate provides the hardware abstraction and synchronization logic for Computed Axial Lithography (CAL) systems.

## Purpose
The success of CAL depends on the precise synchronization between the pre-optimized light projections and the physical rotation of the resin-filled carousel. This crate handles:
- **Motor Control:** Abstraction for rotation stages (e.g., Thorlabs APT stages).
- **Reactive Synchronization:** High-frequency polling of the motor's angular position to select the correct projection frame.
- **Low-Latency Display:** A `wgpu`-based window for presenting optimized projections to the printer's projector.

## Features
- **Thorlabs Motor Support:** Scaffolding for communication with APT-compatible stages via `serialport`.
- **Mock Motor:** A virtual motor implementation for testing the full software pipeline without physical hardware.
- **GPU Projection Window:** Frame-accurate display using textures and `wgpu` to ensure minimal jitter during printing.

## Usage
```rust
use cal_hardware::{MockMotor, ProjectionController};

let motor = MockMotor::new();
let mut controller = ProjectionController::new(motor, optimized_projections, angles);

// Start the sync loop (usually runs in a dedicated thread or winit event loop)
controller.run_sync_loop().unwrap();
```
