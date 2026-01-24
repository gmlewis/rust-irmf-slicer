use bytemuck::{Pod, Zeroable};
use glam::Vec3;

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrimitiveType {
    Sphere = 0,
    Cube = 1,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    Union = 0,
    Difference = 1,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Primitive {
    pub pos: Vec3,
    pub prim_type: u32, // PrimitiveType
    pub size: Vec3,
    pub op: u32, // BooleanOp
}

impl Primitive {
    pub fn new_sphere(pos: Vec3, radius: f32, op: BooleanOp) -> Self {
        Self {
            pos,
            prim_type: PrimitiveType::Sphere as u32,
            size: Vec3::splat(radius),
            op: op as u32,
        }
    }

    pub fn new_cube(pos: Vec3, half_extents: Vec3, op: BooleanOp) -> Self {
        Self {
            pos,
            prim_type: PrimitiveType::Cube as u32,
            size: half_extents,
            op: op as u32,
        }
    }
}
