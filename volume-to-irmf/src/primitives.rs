use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// The type of geometric primitive.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    /// A spherical primitive.
    Sphere = 0,
    /// A cuboid primitive.
    Cube = 1,
}

/// Boolean operation to apply with this primitive.
#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    /// Add this primitive to the shape (union).
    Union = 0,
    /// Subtract this primitive from the shape (difference).
    Difference = 1,
}

/// A geometric primitive used in constructive solid geometry (CSG).
///
/// This struct is designed to be compatible with GPU shaders and uses
/// padding for alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Primitive {
    /// Position of the primitive [x, y, z, padding].
    pub pos: [f32; 4],
    /// Type of primitive (see `PrimitiveType`).
    pub prim_type: u32,
    /// Padding for alignment.
    pub pad1: u32,
    /// Padding for alignment.
    pub pad2: u32,
    /// Padding for alignment.
    pub pad3: u32,
    /// Size of the primitive [width, height, depth, padding].
    pub size: [f32; 4],
    /// Boolean operation (see `BooleanOp`).
    pub op: u32,
    /// Padding for alignment.
    pub pad4: u32,
    /// Padding for alignment.
    pub pad5: u32,
    /// Padding for alignment.
    pub pad6: u32,
}

impl Primitive {
    /// Creates a new spherical primitive.
    ///
    /// # Arguments
    ///
    /// * `pos` - Center position of the sphere.
    /// * `radius` - Radius of the sphere.
    /// * `op` - Boolean operation to apply.
    pub fn new_sphere(pos: Vec3, radius: f32, op: BooleanOp) -> Self {
        Self {
            pos: [pos.x, pos.y, pos.z, 0.0],
            prim_type: PrimitiveType::Sphere as u32,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            size: [radius, radius, radius, 0.0],
            op: op as u32,
            pad4: 0,
            pad5: 0,
            pad6: 0,
        }
    }

    /// Creates a new cuboid primitive.
    ///
    /// # Arguments
    ///
    /// * `pos` - Center position of the cube.
    /// * `half_extents` - Half-extents of the cube in each dimension.
    /// * `op` - Boolean operation to apply.
    pub fn new_cube(pos: Vec3, half_extents: Vec3, op: BooleanOp) -> Self {
        Self {
            pos: [pos.x, pos.y, pos.z, 0.0],
            prim_type: PrimitiveType::Cube as u32,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            size: [half_extents.x, half_extents.y, half_extents.z, 0.0],
            op: op as u32,
            pad4: 0,
            pad5: 0,
            pad6: 0,
        }
    }
}
