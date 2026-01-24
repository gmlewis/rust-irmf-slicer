# volume-to-irmf Implementation Plan

This document outlines the plan for implementing the `volume-to-irmf` package and its associated CLI tools.

## Goal
To convert 3D volume representations (voxels, STLs, etc.) into an optimized IRMF shader (WGSL) that approximates the input model using basic primitives (spheres and cubes) combined with boolean operations.

## Architecture

### 1. `volume-to-irmf` (Library)
The core library providing the optimization engine.

#### Core Components:
- **`VoxelVolume`**: A trait or struct representing the target 3D model.
- **`Primitive`**: A struct representing a geometric primitive (Sphere or Axis-Aligned Cube).
    - Parameters: position (x, y, z), size (radius or half-extents), type (sphere/cube), operation (union/subtraction).
- **`Optimizer`**: The reinforcement learning-like engine.
    - Uses `wgpu` for high-performance error calculation.
    - Implements the "MSE + IoU" hybrid error function.
    - Employs a "divide-and-conquer" or "stochastic search" strategy to refine primitives.
- **`IrmfGenerator`**: Converts the optimized set of primitives into a valid IRMF WGSL shader.

### 2. Supported Input Formats (Executables)
Each executable will use the `volume-to-irmf` library.

- **`binvox-to-irmf`**: Reads `.binvox` files.
- **`dlp-to-irmf`**: Reads `.cbddlp`/`.photon` files.
- **`stl-to-irmf`**: Reads `.stl` files and voxelizes them.
- **`svx-to-irmf`**: Reads `.svx` voxel files.
- **`zip-to-irmf`**: Reads zipped image slices.

## Algorithm Detail: Optimization Loop

1. **Initialization**:
    - Load the target model into a 3D Voxel representation.
    - Upload the target voxels to the GPU as a 3D Texture or Storage Buffer.
    - Start with a single large primitive (either sphere or cube) that roughly fits the bounding box.

2. **Iteration**:
    - **Step A: Stochastic Sampling (on GPU)**:
        - Randomly sample $N$ points within the bounding box.
        - Evaluate the target voxel value at each point.
        - Evaluate the current IRMF approximation (the set of primitives) at each point.
    - **Step B: Error Calculation**:
        - Calculate MSE and IoU for the sampled points.
        - Total Error $L = \alpha \cdot \text{MSE} + \beta \cdot (1 - \text{IoU})$.
    - **Step C: Parameter Refinement (RL-like)**:
        - Perturb primitive parameters (position, size) and see if the error decreases.
        - Use a "divide-and-conquer" approach: if a region has high error, consider splitting a primitive or adding a new one.
        - Handle both Union (adding material) and Difference (removing material) operations.

3. **Termination**:
    - Stop when the maximum error is below a threshold or a maximum number of iterations/primitives is reached.

4. **Generation**:
    - Emit WGSL code that implements the final set of primitives.

## Implementation Steps

### Phase 1: Foundation
- Create the `volume-to-irmf` package directory and `Cargo.toml`.
- Define the `VoxelVolume` and `Primitive` types.
- Implement basic `binvox` and `stl` reading/voxelization.

### Phase 2: WebGPU Optimization Engine
- Set up `wgpu` compute shaders for sampling and error calculation.
- Implement the "Data-driven static compute shader" approach for the candidate model.
- Implement the stochastic refinement loop.

### Phase 3: IRMF Generation
- Implement the logic to turn the list of primitives into WGSL functions.
- Ensure the output follows the IRMF specification (metadata header + shader).

### Phase 4: CLI Tools
- Implement the 5 target executables.
- Add progress bars and logging.

### Phase 5: Testing and Refinement
- Test with standard models (Sphere, Cube, etc.).
- Fine-tune the $\alpha$ and $\beta$ weights for the error function.
- Optimize the GPU sampling for performance.

## Dependencies to Consider
- `wgpu`: For compute shaders.
- `bytemuck`: For data passing to GPU.
- `glam`: For 3D math.
- `image`: For handling image-based voxel slices (zip/svx).
- `anyhow`: For error handling.
- `clap`: For CLI argument parsing.
- `binvox`: (If a good crate exists) for reading binvox.
- `stl`: (If a good crate exists) for reading STL.
