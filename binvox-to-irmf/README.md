# binvox-to-irmf

[![Crates.io](https://img.shields.io/crates/v/binvox-to-irmf.svg)](https://crates.io/crates/binvox-to-irmf)
[![License](https://img.shields.io/crates/l/binvox-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert BinVox voxel files to optimized IRMF shaders.

## Overview

`binvox-to-irmf` reads a 3D voxel model from a BinVox file and converts it into an optimized IRMF (Infinite Resolution Materials Format) shader. It supports both lossless cuboid-merging (default) and Fourier series approximation.

## Installation

```sh
cargo install binvox-to-irmf
```

Or from source:

```sh
cargo install --path binvox-to-irmf
```

## Usage

```sh
binvox-to-irmf input.binvox
```

### Options

- `input`: Input .binvox file
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
