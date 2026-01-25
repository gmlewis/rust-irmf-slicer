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
    pub pos: [f32; 4],
    pub prim_type: u32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
    pub size: [f32; 4],
    pub op: u32,
    pub pad4: u32,
    pub pad5: u32,
    pub pad6: u32,
}

impl Primitive {
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
