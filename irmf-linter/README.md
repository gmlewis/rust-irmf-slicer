# irmf-linter

[![Crates.io](https://img.shields.io/crates/v/irmf-linter.svg)](https://crates.io/crates/irmf-linter)
[![License](https://img.shields.io/crates/l/irmf-linter.svg)](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE)

A linter for IRMF (Infinite Resolution Materials Format) shaders.

This command-line tool validates IRMF files for specification compliance, parsing errors, and include resolution issues.

## Installation

### From crates.io

```sh
cargo install irmf-linter
```

### From source

```sh
git clone https://github.com/gmlewis/rust-irmf-slicer.git
cd rust-irmf-slicer
cargo install --path irmf-linter
```

## Usage

Lint one or more IRMF files:

```sh
irmf-linter file1.irmf file2.irmf
```

The linter will check for:

- **Spec Compliance**: Validates IRMF format requirements
- **Header Parsing**: Ensures valid JSON header structure
- **Include Resolution**: Checks `#include` directives can be resolved
- **Shader Compilation**: Attempts to compile GLSL/WGSL shaders

## Exit Codes

- `0`: All files passed validation
- `1`: One or more validation errors found

## Examples

```sh
# Lint a single file
irmf-linter model.irmf

# Lint multiple files
irmf-linter *.irmf

# Check exit code in scripts
if irmf-linter model.irmf; then
    echo "Model is valid"
else
    echo "Model has errors"
fi
```

## Validation Checks

- Leading `/*{` comment on its own line
- Trailing `}*/` comment on its own line
- Pretty-printed JSON header
- Valid IRMF version and metadata
- Resolvable shader includes
- Compilable shader code

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](https://github.com/gmlewis/rust-irmf-slicer/blob/main/LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please see the main [rust-irmf-slicer](https://github.com/gmlewis/rust-irmf-slicer) repository for details.
