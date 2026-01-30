# dlp-to-irmf

[![Crates.io](https://img.shields.io/crates/v/dlp-to-irmf.svg)](https://crates.io/crates/dlp-to-irmf)
[![License](https://img.shields.io/crates/l/dlp-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert DLP/Photon 3D printing files to optimized IRMF shaders.

## Overview

`dlp-to-irmf` reads a DLP/Photon file (like `.cbddlp` or `.photon`), extracts the voxel layers, and converts them into an optimized IRMF (Infinite Resolution Materials Format) shader. It supports both lossless cuboid-merging (default) and Fourier series approximation.

## Installation

```sh
cargo install dlp-to-irmf
```

Or from source:

```sh
cargo install --path dlp-to-irmf
```

## Usage

```sh
dlp-to-irmf input.cbddlp
```

### Options

- `input`: Input .cbddlp or .photon file
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
