pub mod irmf;
pub mod wgpu_renderer;

pub use irmf::{IrmfModel, IrmfHeader, IrmfError};
pub use wgpu_renderer::WgpuRenderer;
use image::DynamicImage;

pub trait Renderer {
    fn init(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>>;
    fn prepare(&mut self, model: &IrmfModel) -> Result<(), Box<dyn std::error::Error>>;
    fn render(&mut self, z: f32, material_num: usize) -> Result<DynamicImage, Box<dyn std::error::Error>>;
}

pub struct Slicer<R: Renderer> {
    pub model: IrmfModel,
    pub renderer: R,
    pub res_x: f32,
    pub res_y: f32,
    pub res_z: f32,
}

impl<R: Renderer> Slicer<R> {
    pub fn new(model: IrmfModel, renderer: R, res_x: f32, res_y: f32, res_z: f32) -> Self {
        Self {
            model,
            renderer,
            res_x,
            res_y,
            res_z,
        }
    }

    pub fn num_z_slices(&self) -> usize {
        let min_z = self.model.header.min[2];
        let max_z = self.model.header.max[2];
        ((max_z - min_z) / self.res_z).round() as usize
    }

    // Add more slicing methods here...
}
