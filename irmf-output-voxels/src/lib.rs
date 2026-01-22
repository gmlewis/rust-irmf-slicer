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
    pub max_x: f64,
    pub max_y: f64,
    pub max_z: f64,
    pub scale: f64, // Used for standard binvox format
    pub data: Vec<u8>, // Bitset
}

impl BinVox {
    pub fn new(nx: usize, ny: usize, nz: usize, min: [f32; 3], max: [f32; 3]) -> Self {
        let size = (nx * ny * nz + 7) / 8;
        Self {
            nx,
            ny,
            nz,
            min_x: min[0] as f64,
            min_y: min[1] as f64,
            min_z: min[2] as f64,
            max_x: max[0] as f64,
            max_y: max[1] as f64,
            max_z: max[2] as f64,
            scale: (max[2] - min[2]) as f64, // Default to Z scale for compatibility
            data: vec![0; size],
        }
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize) {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return;
        }
        let index = z * self.nx * self.ny + y * self.nx + x;
        self.data[index / 8] |= 1 << (index % 8);
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> bool {
        if x >= self.nx || y >= self.ny || z >= self.nz {
            return false;
        }
        let index = z * self.nx * self.ny + y * self.nx + x;
        (self.data[index / 8] & (1 << (index % 8))) != 0
    }

    pub fn marching_cubes(&self) -> Mesh {
        let mut triangles = Vec::new();
        let dx = (self.max_x - self.min_x) / (self.nx as f64);
        let dy = (self.max_y - self.min_y) / (self.ny as f64);
        let dz = (self.max_z - self.min_z) / (self.nz as f64);

        for z in 0..self.nz {
            for y in 0..self.ny {
                for x in 0..self.nx {
                    if !self.get(x, y, z) {
                        continue;
                    }

                    let fx = self.min_x + (x as f64) * dx;
                    let fy = self.min_y + (y as f64) * dy;
                    let fz = self.min_z + (z as f64) * dz;
                    
                    self.add_cube(&mut triangles, fx as f32, fy as f32, fz as f32, dx as f32, dy as f32, dz as f32, x, y, z);
                }
            }
        }

        Mesh { triangles }
    }

    fn add_cube(&self, tris: &mut Vec<Triangle>, x: f32, y: f32, z: f32, dx: f32, dy: f32, dz: f32, ix: usize, iy: usize, iz: usize) {
        // -X
        if ix == 0 || !self.get(ix - 1, iy, iz) {
            tris.push(Triangle::new([x, y, z], [x, y + dy, z + dz], [x, y, z + dz], [-1.0, 0.0, 0.0]));
            tris.push(Triangle::new([x, y, z], [x, y + dy, z], [x, y + dy, z + dz], [-1.0, 0.0, 0.0]));
        }
        // +X
        if ix == self.nx - 1 || !self.get(ix + 1, iy, iz) {
            tris.push(Triangle::new([x + dx, y, z], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [1.0, 0.0, 0.0]));
            tris.push(Triangle::new([x + dx, y, z], [x + dx, y + dy, z + dz], [x + dx, y + dy, z], [1.0, 0.0, 0.0]));
        }
        // -Y
        if iy == 0 || !self.get(ix, iy - 1, iz) {
            tris.push(Triangle::new([x, y, z], [x, y, z + dz], [x + dx, y, z + dz], [0.0, -1.0, 0.0]));
            tris.push(Triangle::new([x, y, z], [x + dx, y, z + dz], [x + dx, y, z], [0.0, -1.0, 0.0]));
        }
        // +Y
        if iy == self.ny - 1 || !self.get(ix, iy + 1, iz) {
            tris.push(Triangle::new([x, y + dy, z], [x + dx, y + dy, z + dz], [x, y + dy, z + dz], [0.0, 1.0, 0.0]));
            tris.push(Triangle::new([x, y + dy, z], [x + dx, y + dy, z], [x + dx, y + dy, z + dz], [0.0, 1.0, 0.0]));
        }
        // -Z
        if iz == 0 || !self.get(ix, iy, iz - 1) {
            tris.push(Triangle::new([x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [0.0, 0.0, -1.0]));
            tris.push(Triangle::new([x, y, z], [x + dx, y + dy, z], [x, y + dy, z], [0.0, 0.0, -1.0]));
        }
        // +Z
        if iz == self.nz - 1 || !self.get(ix, iy, iz + 1) {
            tris.push(Triangle::new([x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz], [0.0, 0.0, 1.0]));
            tris.push(Triangle::new([x, y, z + dz], [x, y + dy, z + dz], [x + dx, y + dy, z + dz], [0.0, 0.0, 1.0]));
        }
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

#[derive(Copy, Clone, Debug)]
pub struct Triangle {
    pub normal: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub v3: [f32; 3],
}

impl Triangle {
    pub fn new(v1: [f32; 3], v2: [f32; 3], v3: [f32; 3], normal: [f32; 3]) -> Self {
        Self { normal, v1, v2, v3 }
    }
}

pub struct Mesh {
    pub triangles: Vec<Triangle>,
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
