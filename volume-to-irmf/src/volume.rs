use glam::Vec3;

pub struct VoxelVolume {
    pub data: Vec<f32>, // 0.0 to 1.0
    pub dims: [u32; 3],
    pub min: Vec3,
    pub max: Vec3,
}

impl VoxelVolume {
    pub fn new(dims: [u32; 3], min: Vec3, max: Vec3) -> Self {
        let size = (dims[0] * dims[1] * dims[2]) as usize;
        Self {
            data: vec![0.0; size],
            dims,
            min,
            max,
        }
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 0.0;
        }
        let idx = (z * self.dims[1] * self.dims[0] + y * self.dims[0] + x) as usize;
        self.data[idx]
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, val: f32) {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return;
        }
        let idx = (z * self.dims[1] * self.dims[0] + y * self.dims[0] + x) as usize;
        self.data[idx] = val;
    }

    pub fn from_binvox(reader: impl std::io::Read) -> anyhow::Result<Self> {
        use std::io::{BufRead, Read};
        let mut reader = std::io::BufReader::new(reader);
        
        let mut line = String::new();
        reader.read_line(&mut line)?;
        if !line.starts_with("#binvox") {
            anyhow::bail!("Not a binvox file");
        }

        let mut dims = [0u32; 3];
        let mut translate = Vec3::ZERO;
        let mut scale = 1.0f32;

        loop {
            line.clear();
            reader.read_line(&mut line)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() { break; }
            match parts[0] {
                "dim" => {
                    dims[0] = parts[1].parse()?;
                    dims[1] = parts[2].parse()?;
                    dims[2] = parts[3].parse()?;
                }
                "translate" => {
                    translate.x = parts[1].parse()?;
                    translate.y = parts[2].parse()?;
                    translate.z = parts[3].parse()?;
                }
                "scale" => {
                    scale = parts[1].parse()?;
                }
                "data" => break,
                _ => {}
            }
        }

        let mut volume = Self::new(dims, translate, translate + Vec3::splat(scale));
        let total_voxels = (dims[0] * dims[1] * dims[2]) as usize;
        let mut voxels_read = 0;

        while voxels_read < total_voxels {
            let mut pair = [0u8; 2];
            reader.read_exact(&mut pair)?;
            let value = pair[0];
            let count = pair[1] as usize;
            
            let val_f = if value > 0 { 1.0f32 } else { 0.0f32 };
            for _ in 0..count {
                if voxels_read < total_voxels {
                    let x = voxels_read as u32 % dims[0];
                    let z = (voxels_read as u32 / dims[0]) % dims[2];
                    let y = voxels_read as u32 / (dims[0] * dims[2]);
                    volume.set(x, y, z, val_f);
                    voxels_read += 1;
                }
            }
        }

        Ok(volume)
    }

    pub fn from_stl(reader: &mut (impl std::io::Read + std::io::Seek), dims: [u32; 3]) -> anyhow::Result<Self> {
        let mesh = stl_io::read_stl(reader).map_err(|e| anyhow::anyhow!("STL read error: {:?}", e))?;
        
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        
        for v in &mesh.vertices {
            let p = Vec3::new(v[0], v[1], v[2]);
            min = min.min(p);
            max = max.max(p);
        }

        let size = max - min;
        min -= size * 0.05;
        max += size * 0.05;

        let mut volume = Self::new(dims, min, max);
        
        for x in 0..dims[0] {
            for y in 0..dims[1] {
                let px = min.x + (x as f32 + 0.5) / dims[0] as f32 * (max.x - min.x);
                let py = min.y + (y as f32 + 0.5) / dims[1] as f32 * (max.y - min.y);
                
                let mut intersections = Vec::new();
                for tri in &mesh.faces {
                    let v = [
                        mesh.vertices[tri.vertices[0]],
                        mesh.vertices[tri.vertices[1]],
                        mesh.vertices[tri.vertices[2]],
                    ];
                    if let Some(z) = intersect_tri_xy(px, py, &v) {
                        intersections.push(z);
                    }
                }
                
                intersections.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                for i in (0..intersections.len()).step_by(2) {
                    if i + 1 < intersections.len() {
                        let z_start = intersections[i];
                        let z_end = intersections[i+1];
                        
                        let vz_start = (((z_start - min.z) / (max.z - min.z) * dims[2] as f32) as u32).min(dims[2]-1);
                        let vz_end = (((z_end - min.z) / (max.z - min.z) * dims[2] as f32) as u32).min(dims[2]-1);
                        
                        for vz in vz_start..=vz_end {
                            volume.set(x, y, vz, 1.0);
                        }
                    }
                }
            }
        }

        Ok(volume)
    }

    pub fn from_slices(slices: Vec<image::DynamicImage>, min: Vec3, max: Vec3) -> anyhow::Result<Self> {
        if slices.is_empty() {
            anyhow::bail!("No slices provided");
        }
        let width = slices[0].width();
        let height = slices[0].height();
        let depth = slices.len() as u32;
        
        let mut volume = Self::new([width, height, depth], min, max);
        
        for (z, img) in slices.into_iter().enumerate() {
            let gray = img.to_luma8();
            for (x, y, pixel) in gray.enumerate_pixels() {
                volume.set(x, y, z as u32, pixel[0] as f32 / 255.0);
            }
        }
        
        Ok(volume)
    }
}

fn intersect_tri_xy(px: f32, py: f32, v: &[stl_io::Vector<f32>; 3]) -> Option<f32> {
    let v0 = Vec3::new(v[0][0], v[0][1], v[0][2]);
    let v1 = Vec3::new(v[1][0], v[1][1], v[1][2]);
    let v2 = Vec3::new(v[2][0], v[2][1], v[2][2]);
    
    let area = 0.5 * (-v1.y * v2.x + v0.y * (-v1.x + v2.x) + v0.x * (v1.y - v2.y) + v1.x * v2.y);
    if area.abs() < 1e-9 { return None; }
    
    let s = 1.0 / (2.0 * area) * (v0.y * v2.x - v0.x * v2.y + (v2.y - v0.y) * px + (v0.x - v2.x) * py);
    let t = 1.0 / (2.0 * area) * (v0.x * v1.y - v0.y * v1.x + (v0.y - v1.y) * px + (v1.x - v0.x) * py);
    
    if s >= 0.0 && t >= 0.0 && (1.0 - s - t) >= 0.0 {
        Some(v0.z + s * (v1.z - v0.z) + t * (v2.z - v0.z))
    } else {
        None
    }
}