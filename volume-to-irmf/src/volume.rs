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
                    // Order: x changes fastest, then z, then y
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
}
