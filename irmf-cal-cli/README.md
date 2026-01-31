# irmf-cal-cli

A command-line interface for performing Computed Axial Lithography (CAL) directly from IRMF models.

## Purpose
`irmf-cal-cli` brings together `irmf-slicer`, `cal-optimize`, and `cal-hardware` into a single tool for 3D printing. It eliminates the need for voxelization or STL files by treating the IRMF model as an infinite-resolution source for the optimization pipeline.

## Features
- **Direct IRMF Support:** Slice and optimize directly from `.irmf` shader files.
- **Dual Optimization Backends:** Use the GPU for speed or the `--cpu` flag for validation.
- **Adjustable Parameters:** Configure resolution, iteration counts, and learning rates via CLI arguments.

## Usage
```bash
# Optimize an IRMF model using the GPU
irmf-cal-cli --input model.irmf --iterations 20 --res 500

# Optimize using the CPU for comparison
irmf-cal-cli --input model.irmf --cpu --iterations 10
```

## Options
- `-i, --input <FILE>`: Path to the `.irmf` file.
- `-r, --res <MICRONS>`: Desired printing resolution in microns (default: 1000).
- `--iterations <COUNT>`: Number of optimization iterations (default: 10).
- `--cpu`: Force the use of the CPU-based projector.
