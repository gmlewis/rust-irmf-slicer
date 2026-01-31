# Rust [IRMF Shader](https://irmf.io) Slicer

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust CI](https://github.com/gmlewis/rust-irmf-slicer/actions/workflows/rust.yml/badge.svg)](https://github.com/gmlewis/rust-irmf-slicer/actions/workflows/rust.yml)

## Summary

IRMF (Infinite Resolution Materials Format) is a file format used to describe
[GLSL ES](https://en.wikipedia.org/wiki/OpenGL_ES) or
[WGSL](https://www.w3.org/TR/WGSL/) shaders that define the
materials in a 3D object with infinite resolution. IRMF
eliminates the need for traditional [software slicers](https://en.wikipedia.org/wiki/Slicer_(3D_printing)),
[STL](https://en.wikipedia.org/wiki/STL_(file_format)), and
[G-code](https://en.wikipedia.org/wiki/G-code) files used in
[3D printers](https://en.wikipedia.org/wiki/3D_printing).

I believe that IRMF shaders will some day revolutionize the 3D-printing industry.

See [irmf.io](https://irmf.io) for more details.

## About the IRMF Shader Slicer

This Rust crate is a port of the [Go IRMF Slicer](https://github.com/gmlewis/irmf-slicer)
and is a technology demonstration of how to embed an IRMF Slicer into the firmware of
a 3D printer. Included in this repo is a standalone command-line version that demonstrates
the usage of this crate.

The technology stack used is Rust and OpenGL or WebGPU.

This crate can be used by 3D printer hobbyists and manufacturers to natively support
IRMF shaders in addition to G-Code or voxel slices.

The standalone command-line program `irmf_slicer` is a program that
slices an IRMF shader model into either STL files
or into voxel slices (with various output file formats).
For STL files, it outputs one STL file per material.
(Note that some STL files can become enormous, way larger than any online
service bureau currently supports. The resolution can be reduced to limit
the STL file sizes, but at the expense of detail loss.)

For voxel slices, `irmf_slicer` can write them out to ZIP files (one ZIP file per material).
These slices can then be fed to 3D printer software that accepts
voxel slices as input for printing (such as [NanoDLP](https://www.nano3dtech.com/)).

For resin printers using either the [ChiTuBox](https://www.chitubox.com/) or
[AnyCubic](https://store.anycubic.com/collections/resin-3d-printer) slicer
(such as the [Elegoo Mars](https://us.elegoo.com/collections/mars-series)),
the `--dlp` option will output the voxel slices to the `.cbddlp` file
format (which is identical to the `.photon` file format).

Once 3D printers support IRMF shader model files directly for printing,
however, this standalone slicer will no longer be needed.

## LYGIA support

As of 2022-10-27, support has been added for using the LYGIA Shader Library
at: https://lygia.xyz !

This means that you can add lines to your IRMF shaders like this:

```glsl
#include "lygia/math/decimation.glsl"
```

and the source will be retrieved from the LYGIA server.

Congratulations and thanks go to [Patricio Gonzalez Vivo](https://github.com/sponsors/patriciogonzalezvivo)
for making the LYGIA server available for anyone to use, and also
for the amazing tool [glslViewer](https://github.com/patriciogonzalezvivo/glslViewer)!

## Architecture

This project is organized as a Rust workspace to provide a lean core
library suitable for embedding in firmware, while also offering a
full-featured CLI tool.

- **`irmf-slicer`**: The core rendering and slicing library. It is
  designed to be lean, performing no file I/O or networking. It uses
  `wgpu` for hardware-accelerated offscreen rendering.
- **`irmf-slicer-cli`**: A standalone command-line tool for slicing
  IRMF models.
- **`irmf-include-resolver`**: A utility for resolving `#include`
  directives (e.g., from [lygia.xyz](https://lygia.xyz) or GitHub).
- **`irmf-output-stl`**: STL generation logic (using Marching Cubes).
- **`irmf-output-voxels`**: Shared voxel processing and support for
  Binvox, ZIP (PNG slices), Anycubic Photon (.cbddlp), and SVX
  formats.
- **`volume-to-irmf`**: A core library for converting 3D volumes to
  optimized IRMF shaders.
- **`binvox-to-irmf`**, **`stl-to-irmf`**, **`obj-to-irmf`**,
  **`svx-to-irmf`**, **`zip-to-irmf`**: CLI tools for converting
  various 3D formats into IRMF models.

## Computed Axial Lithography (CAL)

This project now includes a complete Rust port of the Computed Axial
Lithography (CAL) software, originally developed by the
Prof. Hayden Taylor Lab at UC Berkeley.

CAL is a high-speed 3D printing method that uses a rotating volume of
photosensitive resin. By projecting a sequence of optimized 2D images
through the rotating volume, the cumulative light dose solidifies the
entire 3D object at once.

- **`cal-optimize`**: The core optimization library. It uses iterative
  gradient descent (Radon and Inverse Radon transforms) to find the
  optimal set of projections for a given IRMF model. It supports both
  GPU (`wgpu`) and CPU backends.
- **`cal-hardware`**: Hardware abstraction for rotation stages (e.g.,
  Thorlabs APT) and real-time projection synchronization.
- **`irmf-cal-cli`**: A standalone tool for performing IRMF-based CAL
  optimization and printing.

### CAL Usage

Optimize an IRMF model using the GPU:
```sh
cargo run -p irmf-cal-cli -- --input examples/001-sphere/sphere-cal.irmf --iterations 20
```

Force CPU optimization for validation:
```sh
cargo run -p irmf-cal-cli -- --input examples/001-sphere/sphere-cal.irmf --cpu
```

### Production Workflow

**1. Mock Print (No Hardware):**

Use this mode to test the visualization and synchronization on your
local machine. A window will pop up showing the projections as the
virtual motor rotates.

```sh
cargo run -p irmf-cal-cli -- --input examples/001-sphere/sphere-cal.irmf --mock --iterations 20
```

**2. Physical Print (DLP Projector + Thorlabs Stage):**

Use this mode in the lab to drive the actual 3D printer. The
projections will be displayed borderless on the selected monitor.

```sh
cargo run -p irmf-cal-cli -- --input examples/001-sphere/sphere-cal.irmf --port /dev/ttyUSB0 --monitor 1 --iterations 50
```

## Features

- **Multi-Language Support**: Full support for both GLSL and WGSL shaders.
- **Hardware Acceleration**: Uses `wgpu` for fast, cross-platform GPU rendering (Vulkan, Metal, DX12, WebGPU).
- **Multiple Output Formats**:
  - **STL**: High-quality meshes generated via Marching Cubes.
  - **Binvox**: Standard voxel format.
  - **ZIP**: Archive of PNG slices, compatible with [NanoDLP](https://www.nano3dtech.com/).
  - **DLP (.cbddlp)**: Compatible with Anycubic and ChiTuBox-based resin printers.
  - **SVX**: Simple Voxel Format.
- **Remote Includes**: Automatic resolution of `#include` directives from `lygia.xyz` and GitHub.
- **Offscreen Rendering**: No windowing system required for slicing (though a `--view` mode is available for debugging).

## How it works

This slicer dicing up your model (the IRMF shader) into slices (planes)
that are perpendicular (normal) to the Z (up) axis. The slices are very
thin and when stacked together, represent your solid model.

Using the `--zip` option, the result is one ZIP file per model material
with all the slices in the root of the ZIP so as to be compatible
with NanoDLP. When using the `--zip` option, the resolution is set
to X: 65, Y: 60, Z: 30 microns (unless the `--res` option is used to
override this) in order to support the `MCAST + Sylgard / 65 micron`
option of NanoDLP.

Using the `--dlp` option, the result is one `.cbddlp` file per model material
that can be loaded into the [ChiTuBox](https://www.chitubox.com/) or
[AnyCubic](https://store.anycubic.com/collections/resin-3d-printer)
slicer directly (`.cbddlp` is identical to the `.photon` file format).

Using the `--stl` option, the result is one STL file per model material.

Using the `--binvox` option, it will write one `.binvox` file per model material.

## Usage (CLI)

### Installation

After you have a recent version of [Rust](https://rustup.rs/) installed,
run the following command in a terminal window:

```sh
$ cargo install --path irmf-slicer-cli
```

(Or `$ cargo install irmf-slicer` if you are installing from [crates.io](https://crates.io).)

### Examples

To slice one or more `.irmf` files, just list them on the command line.

Slice a model into STL files:
```sh
irmf-slicer-cli --stl examples/001-sphere/sphere-cal.irmf
```

Slice a model for a resin printer (DLP):
```sh
irmf-slicer-cli --dlp examples/001-sphere/sphere-cal.irmf
```

View the slicing process in real-time:
```sh
irmf-slicer-cli --view --zip examples/002-cube/cube-1.irmf
```

The output files will be saved in the same directory as the original
input IRMF files.

## Usage (Library)

To use the IRMF slicer in your own Rust project, add the `irmf-slicer` crate to your `Cargo.toml`.

```rust
use irmf_slicer::{IrmfModel, WgpuRenderer, Slicer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read("model.irmf")?;
    let model = IrmfModel::new(&data)?;
    
    let renderer = WgpuRenderer::new().await?;
    let mut slicer = Slicer::new(model, renderer, 42.0, 42.0, 42.0);
    
    slicer.prepare_render_z()?;
    slicer.render_z_slices(1, |idx, z, radius, img| {
        img.save(format!("slice_{:04}.png", idx))?;
        Ok(())
    })?;
    
    Ok(())
}
```

## Usage (Conversion)

The `volume-to-irmf` tools allow you to convert existing 3D models into optimized IRMF shaders. This process can either perform a lossless cuboid-merging optimization (default) or a Fourier series approximation for extreme compression and infinite resolution.

### Examples

Convert a BinVox file to IRMF (lossless cuboids):
```sh
cargo run -p binvox-to-irmf -- input.binvox
```

Convert an STL file to IRMF using Fourier approximation (specifying resolution and coefficients):
```sh
cargo run -p stl-to-irmf -- input.stl --res 128 --fourier -k 16
```

Convert an OBJ file to IRMF on GPU:
```sh
cargo run -p obj-to-irmf -- input.obj --res 128 --gpu
```

Convert a ZIP archive of image slices to IRMF:
```sh
cargo run -p zip-to-irmf -- input.zip
```

These tools will output a `.irmf` file containing an optimized shader that represents the model.

## License

Copyright 2019 Glenn M. Lewis. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
