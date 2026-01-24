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
}
