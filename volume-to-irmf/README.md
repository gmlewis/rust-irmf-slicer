# volume-to-irmf

[![Crates.io](https://img.shields.io/crates/v/volume-to-irmf.svg)](https://crates.io/crates/volume-to-irmf)
[![Documentation](https://docs.rs/volume-to-irmf/badge.svg)](https://docs.rs/volume-to-irmf)
[![License](https://img.shields.io/crates/l/volume-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A library for converting 3D volumes to optimized IRMF shaders using reinforcement learning.

This crate provides tools to convert voxel-based 3D volumes into efficient IRMF (Infinite Resolution Materials Format) shaders through constructive solid geometry (CSG) optimization.

## Features

- **Lossless Optimization**: Converts voxel data to optimized cuboid representations
- **GPU Acceleration**: Uses WGPU for high-performance computations
- **IRMF Generation**: Produces optimized GLSL/WGSL shaders
- **Multiple Formats**: Supports BinVox, STL, OBJ, DLP/Photon, and other voxel formats

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
volume-to-irmf = "0.2"
```

### Basic Example

```rust
use volume_to_irmf::{Optimizer, VoxelVolume};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load a voxel volume
    let volume = VoxelVolume::from_binvox(std::fs::File::open("model.binvox")?)?;

    // Create and run optimizer
    let mut optimizer = Optimizer::new(volume).await?;
    optimizer.run_lossless().await?;

    // Generate IRMF shader
    let irmf_code = optimizer.generate_final_irmf("glsl".to_string());
    println!("{}", irmf_code);

    Ok(())
}
```

## API Documentation

Full API documentation is available at [docs.rs/volume-to-irmf](https://docs.rs/volume-to-irmf).

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please see the main [rust-irmf-slicer](https://github.com/gmlewis/rust-irmf-slicer) repository for details.
