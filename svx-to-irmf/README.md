# svx-to-irmf

[![Crates.io](https://img.shields.io/crates/v/svx-to-irmf.svg)](https://crates.io/crates/svx-to-irmf)
[![License](https://img.shields.io/crates/l/svx-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert SVX (Simple Voxel Format) files to optimized IRMF shaders.

## Overview

`svx-to-irmf` reads a 3D voxel model from an SVX file and converts it into an optimized IRMF (Infinite Resolution Materials Format) shader. It supports both lossless cuboid-merging (default) and Fourier series approximation.

## Installation

```sh
cargo install svx-to-irmf
```

Or from source:

```sh
cargo install --path svx-to-irmf
```

## Usage

```sh
svx-to-irmf input.svx
```

### Options

- `input`: Input .svx file
- `-o, --output`: Output .irmf file (optional, defaults to input name with .irmf extension)
- `-l, --language`: IRMF shader language (glsl or wgsl, defaults to glsl)
- `--gpu`: Use GPU for optimization
- `--fourier`: Use Fourier approximation instead of cuboid merging
- `-k`: Number of Fourier coefficients in each dimension (defaults to 16)
- `--pass2`: Save intermediate Pass 2 (X-runs) debug IRMF
- `--pass3`: Save intermediate Pass 3 (XY-planes) debug IRMF

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
