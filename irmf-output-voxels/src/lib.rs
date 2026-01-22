pub mod zip_out;
pub mod binvox_out;

use std::io::{Write};

pub struct BinVox {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub min_x: f64,
    pub min_y: f64,
    pub min_z: f64,
    pub scale: f64,
    pub data: Vec<u8>, // Bitset
}

impl BinVox {
    pub fn new(nx: usize, ny: usize, nz: usize, min_x: f64, min_y: f64, min_z: f64, scale: f64) -> Self {
        let size = (nx * ny * nz + 7) / 8;
        Self {
            nx,
            ny,
            nz,
            min_x,
            min_y,
            min_z,
            scale,
            data: vec![0; size],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize) {
        let index = z * self.nx * self.ny + y * self.nx + x;
        if index / 8 < self.data.len() {
            self.data[index / 8] |= 1 << (index % 8);
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> bool {
        let index = z * self.nx * self.ny + y * self.nx + x;
        if index / 8 < self.data.len() {
            (self.data[index / 8] & (1 << (index % 8))) != 0
        } else {
            false
        }
    }

    pub fn marching_cubes(&self) -> Mesh {
        Mesh { triangles: Vec::new() }
    }

    pub fn write_binvox<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        writeln!(w, "#binvox 1")?;
        writeln!(w, "dim {} {} {}", self.nx, self.ny, self.nz)?;
        writeln!(w, "translate {} {} {}", self.min_x, self.min_y, self.min_z)?;
        writeln!(w, "scale {}", self.scale)?;
        writeln!(w, "data")?;

        let mut current_value = self.get(0, 0, 0);
        let mut count = 0u8;

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    let val = self.get(x, y, z);
                    if val == current_value && count < 255 {
                        count += 1;
                    } else {
                        w.write_all(&[current_value as u8, count])?;
                        current_value = val;
                        count = 1;
                    }
                }
            }
        }
        w.write_all(&[current_value as u8, count])?;
        Ok(())
    }
}

pub struct Mesh {
    pub triangles: Vec<Triangle>,
}

pub struct Triangle {
    pub normal: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub v3: [f32; 3],
}

impl Mesh {
    pub fn save_stl<W: Write>(&self, mut w: W) -> std::io::Result<()> {
        let header = [0u8; 80];
        w.write_all(&header)?;
        let count = self.triangles.len() as u32;
        w.write_all(&count.to_le_bytes())?;

        for tri in &self.triangles {
            w.write_all(bytemuck::cast_slice(&tri.normal))?;
            w.write_all(bytemuck::cast_slice(&tri.v1))?;
            w.write_all(bytemuck::cast_slice(&tri.v2))?;
            w.write_all(bytemuck::cast_slice(&tri.v3))?;
            w.write_all(&0u16.to_le_bytes())?;
        }
        Ok(())
    }
}
