# stl-to-irmf

[![Crates.io](https://img.shields.io/crates/v/stl-to-irmf.svg)](https://crates.io/crates/stl-to-irmf)
[![License](https://img.shields.io/crates/l/stl-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert STL mesh files to optimized IRMF shaders.

## Overview

`stl-to-irmf` voxelizes a 3D mesh from an STL file and then converts it into an optimized IRMF (Infinite Resolution Materials Format) shader. It supports both lossless cuboid-merging (default) and Fourier series approximation.

## Installation

```sh
cargo install stl-to-irmf
```

Or from source:

```sh
cargo install --path stl-to-irmf
```

## Usage

```sh
stl-to-irmf --res 128 input.stl
```

### Options

- `input`: Input .stl file
- `-o, --output`: Output .irmf file (optional, defaults to input name with .irmf extension)
- `-l, --language`: IRMF shader language (glsl or wgsl, defaults to glsl)
- `-r, --res`: Resolution for voxelization (defaults to 64)
- `--gpu`: Use GPU for voxelization and optimization
- `--fourier`: Use Fourier approximation instead of cuboid merging
- `-k`: Number of Fourier coefficients in each dimension (defaults to 16)
- `--debug`: Dump debug information to stdout
- `--pass2`: Save intermediate Pass 2 (X-runs) debug IRMF
- `--pass3`: Save intermediate Pass 3 (XY-planes) debug IRMF

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
