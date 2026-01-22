//! Mock implementation of the IRMF renderer for testing.

use crate::irmf::IrmfModel;
use crate::{IrmfResult, Renderer};
use image::{DynamicImage, RgbaImage};

/// A mock renderer that does not require a GPU.
///
/// It generates simple placeholder images (e.g., a solid color or a gradient)
/// instead of actually executing the shader.
pub struct MockRenderer {
    pub width: u32,
    pub height: u32,
}

impl MockRenderer {
    /// Creates a new `MockRenderer` instance.
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
        }
    }
}

impl Default for MockRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl Renderer for MockRenderer {
    fn init(&mut self, width: u32, height: u32) -> IrmfResult<()> {
        self.width = width;
        self.height = height;
        Ok(())
    }

    fn prepare(
        &mut self,
        _model: &IrmfModel,
        _vertices: &[f32],
        _projection: glam::Mat4,
        _camera: glam::Mat4,
        _model_matrix: glam::Mat4,
        _vec3_str: &str,
    ) -> IrmfResult<()> {
        Ok(())
    }

    fn render(&mut self, _slice_depth: f32, _material_num: usize) -> IrmfResult<DynamicImage> {
        let mut img = RgbaImage::new(self.width, self.height);
        // Generate a simple pattern: a solid white pixel if we "think" we are inside a 10mm sphere
        // This is just a mock, so we don't need real shader logic.
        for y in 0..self.height {
            for x in 0..self.width {
                img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            }
        }
        Ok(DynamicImage::ImageRgba8(img))
    }
}
