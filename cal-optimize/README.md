# cal-optimize

This crate provides the core mathematical optimization logic for Computed Axial Lithography (CAL).

## Origin
The algorithms in this crate are ported from the [CAL-software-Matlab](https://github.com/computed-axial-lithography/CAL-software-Matlab) project, specifically from `CALOptimize.m` and `CALProjectorConstructor.m`.

## Purpose
Computed Axial Lithography works by finding an optimal set of 2D projections that, when integrated over a full rotation, reconstruct a target 3D volume. This crate implements:
- **Forward Projector (Radon Transform):** Simulates the light dose from a set of 2D images.
- **Backward Projector (Back-projection):** Reconstructs a 3D volume from 2D projections.
- **Iterative Optimizer:** A gradient descent loop that minimizes the error between the simulated reconstruction and the target volume, incorporating the non-linear "sigmoid" feedback that models the resin's chemical threshold.

## Features
- **GPU Acceleration:** High-performance projections using `wgpu` compute shaders (Radon and Inverse Radon transforms).
- **CPU Fallback:** A pure-Rust implementation using `ndarray` for systems without compatible GPUs.
- **IRMF Integration:** Directly uses `irmf-slicer` to generate the 3D target volume from Infinite Resolution Materials Format (IRMF) models.

## Usage
```rust
use cal_optimize::{CalOptimizer, GpuProjector, TargetVolume};
use irmf_slicer::{irmf::IrmfModel, Slicer, wgpu_renderer::WgpuRenderer};

// ... set up slicer ...
let target = TargetVolume::from_irmf(&mut slicer, 0).unwrap();
let angles: Vec<f32> = (0..180).map(|a| a as f32).collect();

let projector = GpuProjector::new().await;
let mut optimizer = CalOptimizer::new(target, angles, projector);
let (optimized_projections, error_history) = optimizer.run();
```
