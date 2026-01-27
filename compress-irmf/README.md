# compress-irmf

[![Crates.io](https://img.shields.io/crates/v/compress-irmf.svg)](https://crates.io/crates/compress-irmf)
[![License](https://img.shields.io/crates/l/compress-irmf.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A tool to compress IRMF (Infinite Resolution Materials Format) shaders.

This command-line utility compresses IRMF shader payloads using gzip compression and optional base64 encoding, reducing file sizes for storage and transmission.

## Installation

### From crates.io

```sh
cargo install compress-irmf
```

### From source

```sh
git clone https://github.com/gmlewis/rust-irmf-slicer.git
cd rust-irmf-slicer
cargo install --path compress-irmf
```

## Usage

Compress an IRMF file:

```sh
compress-irmf input.irmf
```

This creates `input.irmf.compressed.irmf` with gzip-compressed shader data.

### Options

- `-o, --output <FILE>`: Specify output file path (default: `<input>.compressed.irmf`)
- `-b, --base64`: Use base64 encoding in addition to gzip compression

### Examples

Compress with base64 encoding:

```sh
compress-irmf --base64 model.irmf
```

Specify custom output:

```sh
compress-irmf -o compressed.irmf model.irmf
```

## Compression Details

- **Gzip**: Standard DEFLATE compression for shader source code
- **Base64 + Gzip**: Additional base64 encoding for environments that require text-only data
- **Automatic Header Update**: Updates IRMF header with appropriate encoding field

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please see the main [rust-irmf-slicer](https://github.com/gmlewis/rust-irmf-slicer) repository for details.
