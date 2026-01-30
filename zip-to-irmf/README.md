# zip-to-irmf

[![Crates.io](https://img.shields.io/crates/v/zip-to-irmf.svg)](https://crates.io/crates/zip-to-irmf)
[![License](https://img.shields.io/crates/l/zip-to-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A command-line tool to convert ZIP archives of image slices to optimized IRMF shaders.

## Overview

`zip-to-irmf` reads a ZIP file containing image slices (PNG, JPG, or BMP), reconstructs the 3D volume, and converts it into an optimized IRMF (Infinite Resolution Materials Format) shader. It supports both lossless cuboid-merging (default) and Fourier series approximation.

## Installation

```sh
cargo install zip-to-irmf
```

Or from source:

```sh
cargo install --path zip-to-irmf
```

## Usage

```sh
zip-to-irmf input.zip
```

### Options

- `input`: Input .zip file
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
