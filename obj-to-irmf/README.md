# obj-to-irmf

[![Crates.io](https://img.shields.io/crates/v/obj-to-irmf.svg)](https://crates.io/crates/obj-to-irmf)
[![License](https://img.shields.io/crates/l/obj-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert Wavefront OBJ files to optimized IRMF shaders.

## Overview

`obj-to-irmf` voxelizes a 3D mesh from an OBJ file and then uses a lossless cuboid-merging algorithm to generate an optimized IRMF (Infinite Resolution Materials Format) shader. This shader represents the 3D model efficiently and can be used with the IRMF slicer.

## Installation

```sh
cargo install obj-to-irmf
```

Or from source:

```sh
cargo install --path obj-to-irmf
```

## Usage

```sh
obj-to-irmf --res 128 input.obj
```

### Options

- `input`: Input .obj file
- `-o, --output`: Output .irmf file (optional, defaults to input name with .irmf extension)
- `-l, --language`: IRMF shader language (glsl or wgsl, defaults to glsl)
- `-r, --res`: Resolution for voxelization (defaults to 64)
- `--gpu`: Use GPU for voxelization and optimization
- `--fourier`: Use Fourier approximation instead of cuboid merging
- `-k`: Number of Fourier coefficients in each dimension (defaults to 16)
- `--debug`: Dump debug information to stdout

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
