//! volume-to-irmf
//!
//! A library for converting 3D volumes to optimized IRMF shaders using reinforcement learning.

pub mod optimizer;
pub mod primitives;
pub mod volume;

pub use optimizer::Optimizer;
pub use primitives::Primitive;
pub use volume::VoxelVolume;
