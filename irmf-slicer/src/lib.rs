//! Core library for the IRMF slicer.
//!
//! This crate provides the core logic for parsing, rendering, and slicing
//! Infinite Resolution Materials Format (IRMF) models.

pub mod irmf;
pub mod mock_renderer;
pub mod wgpu_renderer;

pub use image::DynamicImage;
pub use irmf::{IrmfError, IrmfHeader, IrmfModel};
pub use mock_renderer::MockRenderer;
pub use wgpu_renderer::WgpuRenderer;

/// Result type for IRMF operations.
pub type IrmfResult<T> = Result<T, IrmfError>;

/// Trait defining the interface for an IRMF renderer.
pub trait Renderer {
    /// Initializes the renderer with the given dimensions.
    fn init(&mut self, width: u32, height: u32) -> IrmfResult<()>;

    /// Prepares the renderer for slicing a model.
    ///
    /// # Arguments
    ///
    /// * `model` - The IRMF model to render.
    /// * `vertices` - The vertices for the full-screen quad (or similar) used for rendering.
    /// * `projection` - The projection matrix.
    /// * `camera` - The camera view matrix.
    /// * `model_matrix` - The model transformation matrix.
    /// * `vec3_str` - A string representing how to construct the `vec3` position in the shader.
    fn prepare(
        &mut self,
        model: &IrmfModel,
        vertices: &[f32],
        projection: glam::Mat4,
        camera: glam::Mat4,
        model_matrix: glam::Mat4,
        vec3_str: &str,
    ) -> IrmfResult<()>;

    /// Renders a single slice of the model.
    ///
    /// # Arguments
    ///
    /// * `slice_depth` - The depth (Z-coordinate, or relevant axis) of the slice.
    /// * `material_num` - The index of the material to render.
    fn render(&mut self, slice_depth: f32, material_num: usize) -> IrmfResult<DynamicImage>;
}

/// A slicer that orchestrates the rendering of multiple slices of an IRMF model.
pub struct Slicer<R: Renderer> {
    /// The IRMF model being sliced.
    pub model: IrmfModel,
    /// The renderer used to generate slice images.
    pub renderer: R,
    /// Resolution in the X dimension (microns).
    pub res_x: f32,
    /// Resolution in the Y dimension (microns).
    pub res_y: f32,
    /// Resolution in the Z dimension (microns).
    pub res_z: f32,
}

impl<R: Renderer> Slicer<R> {
    /// Creates a new `Slicer` instance.
    pub fn new(model: IrmfModel, renderer: R, res_x: f32, res_y: f32, res_z: f32) -> Self {
        Self {
            model,
            renderer,
            res_x,
            res_y,
            res_z,
        }
    }

    /// Returns the number of slices in the X dimension.
    pub fn num_x_slices(&self) -> usize {
        let delta_x = self.res_x / 1000.0;
        let min_x = self.model.header.min[0];
        let max_x = self.model.header.max[0];
        let mut n = (0.5 + (max_x - min_x) / delta_x).floor() as usize;
        if n % 2 == 1 {
            n += 1;
        }
        n
    }

    /// Returns the number of slices in the Y dimension.
    pub fn num_y_slices(&self) -> usize {
        let nx = self.num_x_slices();
        let delta_y = self.res_y / 1000.0;
        let min_y = self.model.header.min[1];
        let max_y = self.model.header.max[1];
        let mut n = (0.5 + (max_y - min_y) / delta_y).floor() as usize;
        if nx % 2 == 1 {
            n += 1;
        }
        n
    }

    /// Returns the number of slices in the Z dimension.
    pub fn num_z_slices(&self) -> usize {
        let delta_z = self.res_z / 1000.0;
        let min_z = self.model.header.min[2];
        let max_z = self.model.header.max[2];
        (0.5 + (max_z - min_z) / delta_z).floor() as usize
    }

    /// Prepares the renderer for slicing in the X dimension.
    pub fn prepare_render_x(&mut self) -> IrmfResult<()> {
        let left = self.model.header.min[1];
        let right = self.model.header.max[1];
        let bottom = self.model.header.min[2];
        let top = self.model.header.max[2];

        let delta_y = self.res_y / 1000.0;
        let delta_z = self.res_z / 1000.0;

        let aspect_ratio = ((right - left) * delta_z) / ((top - bottom) * delta_y);
        let mut new_width = (0.5 + (right - left) / delta_y).floor() as u32;
        let mut new_height = (0.5 + (top - bottom) / delta_z).floor() as u32;

        if aspect_ratio * (new_height as f32) < (new_width as f32) {
            new_height = (0.5 + (new_width as f32) / aspect_ratio).floor() as u32;
        }

        if new_width % 2 == 1 {
            new_width += 1;
            new_height += 1;
        }

        self.renderer.init(new_width, new_height)?;

        let vertices = [
            // Tri 1
            0.0, left, bottom, 0.0, right, bottom, 0.0, left, top, // Tri 2
            0.0, right, bottom, 0.0, right, top, 0.0, left, top,
        ];

        let near = 0.1;
        let far = 100.0;
        let projection = glam::Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let camera = glam::Mat4::look_at_rh(
            glam::vec3(3.0, 0.0, 0.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 0.0, 1.0),
        );
        let model_matrix = glam::Mat4::IDENTITY;
        let vec3_str = "u_slice, fragVert.yz";

        self.renderer.prepare(
            &self.model,
            &vertices,
            projection,
            camera,
            model_matrix,
            vec3_str,
        )
    }

    /// Prepares the renderer for slicing in the Y dimension.
    pub fn prepare_render_y(&mut self) -> IrmfResult<()> {
        let left = self.model.header.min[0];
        let right = self.model.header.max[0];
        let bottom = self.model.header.min[2];
        let top = self.model.header.max[2];

        let delta_x = self.res_x / 1000.0;
        let delta_z = self.res_z / 1000.0;

        let aspect_ratio = ((right - left) * delta_z) / ((top - bottom) * delta_x);
        let mut new_width = (0.5 + (right - left) / delta_x).floor() as u32;
        let mut new_height = (0.5 + (top - bottom) / delta_z).floor() as u32;

        if aspect_ratio * (new_height as f32) < (new_width as f32) {
            new_height = (0.5 + (new_width as f32) / aspect_ratio).floor() as u32;
        }

        if new_width % 2 == 1 {
            new_width += 1;
            new_height += 1;
        }

        self.renderer.init(new_width, new_height)?;

        let vertices = [
            // Tri 1
            left, 0.0, bottom, right, 0.0, bottom, left, 0.0, top, // Tri 2
            right, 0.0, bottom, right, 0.0, top, left, 0.0, top,
        ];

        let near = 0.1;
        let far = 100.0;
        let projection = glam::Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let camera = glam::Mat4::look_at_rh(
            glam::vec3(0.0, -3.0, 0.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 0.0, 1.0),
        );
        let model_matrix = glam::Mat4::IDENTITY;
        let vec3_str = "fragVert.x, u_slice, fragVert.z";

        self.renderer.prepare(
            &self.model,
            &vertices,
            projection,
            camera,
            model_matrix,
            vec3_str,
        )
    }

    /// Prepares the renderer for slicing in the Z dimension.
    pub fn prepare_render_z(&mut self) -> IrmfResult<()> {
        let left = self.model.header.min[0];
        let right = self.model.header.max[0];
        let bottom = self.model.header.min[1];
        let top = self.model.header.max[1];

        let delta_x = self.res_x / 1000.0;
        let delta_y = self.res_y / 1000.0;

        let aspect_ratio = ((right - left) * delta_y) / ((top - bottom) * delta_x);
        let mut new_width = (0.5 + (right - left) / delta_x).floor() as u32;
        let mut new_height = (0.5 + (top - bottom) / delta_y).floor() as u32;

        if aspect_ratio * (new_height as f32) < (new_width as f32) {
            new_height = (0.5 + (new_width as f32) / aspect_ratio).floor() as u32;
        }

        if new_width % 2 == 1 {
            new_width += 1;
            new_height += 1;
        }

        self.renderer.init(new_width, new_height)?;

        let vertices = [
            // Tri 1
            left, bottom, 0.0, right, bottom, 0.0, left, top, 0.0, // Tri 2
            right, bottom, 0.0, right, top, 0.0, left, top, 0.0,
        ];

        let near = 0.1;
        let far = 100.0;
        let projection = glam::Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let camera = glam::Mat4::look_at_rh(
            glam::vec3(0.0, 0.0, 3.0),
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(0.0, 1.0, 0.0),
        );
        let model_matrix = glam::Mat4::IDENTITY;
        let vec3_str = "fragVert.xy, u_slice";

        self.renderer.prepare(
            &self.model,
            &vertices,
            projection,
            camera,
            model_matrix,
            vec3_str,
        )
    }

    /// Renders a single slice in the X dimension.
    pub fn render_x_slice(
        &mut self,
        slice_num: usize,
        material_num: usize,
    ) -> IrmfResult<DynamicImage> {
        let delta_x = self.res_x / 1000.0;
        let voxel_radius_x = 0.5 * delta_x;
        let min_x = self.model.header.min[0];
        let slice_depth = min_x + voxel_radius_x + (slice_num as f32) * delta_x;
        self.renderer.render(slice_depth, material_num)
    }

    /// Renders a single slice in the Y dimension.
    pub fn render_y_slice(
        &mut self,
        slice_num: usize,
        material_num: usize,
    ) -> IrmfResult<DynamicImage> {
        let delta_y = self.res_y / 1000.0;
        let voxel_radius_y = 0.5 * delta_y;
        let min_y = self.model.header.min[1];
        let slice_depth = min_y + voxel_radius_y + (slice_num as f32) * delta_y;
        self.renderer.render(slice_depth, material_num)
    }

    /// Renders a single slice in the Z dimension.
    pub fn render_z_slice(
        &mut self,
        slice_num: usize,
        material_num: usize,
    ) -> IrmfResult<DynamicImage> {
        let delta_z = self.res_z / 1000.0;
        let voxel_radius_z = 0.5 * delta_z;
        let min_z = self.model.header.min[2];
        let slice_depth = min_z + voxel_radius_z + (slice_num as f32) * delta_z;
        self.renderer.render(slice_depth, material_num)
    }

    /// Iteratively renders all slices in the X dimension and calls a callback for each.
    pub fn render_x_slices<F>(&mut self, material_num: usize, mut f: F) -> IrmfResult<()>
    where
        F: FnMut(usize, f32, f32, DynamicImage) -> IrmfResult<()>,
    {
        let num_slices = self.num_x_slices();
        let delta_x = self.res_x / 1000.0;
        let voxel_radius_x = 0.5 * delta_x;
        let min_x = self.model.header.min[0];

        for n in 0..num_slices {
            let x = min_x + voxel_radius_x + (n as f32) * delta_x;
            let img = self.renderer.render(x, material_num)?;
            f(n, x, voxel_radius_x, img)?;
        }

        Ok(())
    }

    /// Iteratively renders all slices in the Y dimension and calls a callback for each.
    pub fn render_y_slices<F>(&mut self, material_num: usize, mut f: F) -> IrmfResult<()>
    where
        F: FnMut(usize, f32, f32, DynamicImage) -> IrmfResult<()>,
    {
        let num_slices = self.num_y_slices();
        let delta_y = self.res_y / 1000.0;
        let voxel_radius_y = 0.5 * delta_y;
        let min_y = self.model.header.min[1];

        for n in 0..num_slices {
            let y = min_y + voxel_radius_y + (n as f32) * delta_y;
            let img = self.renderer.render(y, material_num)?;
            f(n, y, voxel_radius_y, img)?;
        }

        Ok(())
    }

    /// Iteratively renders all slices in the Z dimension and calls a callback for each.
    pub fn render_z_slices<F>(&mut self, material_num: usize, mut f: F) -> IrmfResult<()>
    where
        F: FnMut(usize, f32, f32, DynamicImage) -> IrmfResult<()>,
    {
        let num_slices = self.num_z_slices();
        let delta_z = self.res_z / 1000.0;
        let voxel_radius_z = 0.5 * delta_z;
        let min_z = self.model.header.min[2];

        for n in 0..num_slices {
            let z = min_z + voxel_radius_z + (n as f32) * delta_z;
            let img = self.renderer.render(z, material_num)?;
            f(n, z, voxel_radius_z, img)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock_renderer::MockRenderer;
    use glam::Vec4Swizzles;

    #[test]
    fn test_slicer_num_slices() {
        let data = b"/*{\"irmf\":\"1.0\",\"materials\":[\"PLA\"],\"max\":[5,5,5],\"min\":[-5,-5,-5],\"units\":\"mm\"}*/\nvoid mainModel4(out vec4 materials, in vec3 xyz) { materials[0] = 1.0; }";
        let model = IrmfModel::new(data).unwrap();
        let renderer = MockRenderer::new();
        let slicer = Slicer::new(model, renderer, 1000.0, 1000.0, 1000.0);
        assert_eq!(slicer.num_x_slices(), 10);
        assert_eq!(slicer.num_y_slices(), 10);
        assert_eq!(slicer.num_z_slices(), 10);
    }

    #[test]
    fn test_slicer_prepare_render() {
        let data = b"/*{\"irmf\":\"1.0\",\"materials\":[\"PLA\"],\"max\":[5,5,5],\"min\":[-5,-5,-5],\"units\":\"mm\"}*/\nvoid mainModel4(out vec4 materials, in vec3 xyz) { materials[0] = 1.0; }";
        let model = IrmfModel::new(data).unwrap();
        let renderer = MockRenderer::new();
        let mut slicer = Slicer::new(model, renderer, 1000.0, 1000.0, 1000.0);
        slicer.prepare_render_z().unwrap();
        assert_eq!(slicer.renderer.width, 10);
        assert_eq!(slicer.renderer.height, 10);
    }

    #[test]
    fn test_coordinate_mapping() {
        let data = b"/*{\"irmf\":\"1.0\",\"materials\":[\"PLA\"],\"max\":[5,5,5],\"min\":[-5,-5,-5],\"units\":\"mm\"}*/\nvoid mainModel4(out vec4 materials, in vec3 xyz) { materials[0] = 1.0; }";
        let model = IrmfModel::new(data).unwrap();
        let renderer = MockRenderer::new();
        let mut slicer = Slicer::new(model, renderer, 1000.0, 1000.0, 1000.0);

        // Test Z-slicing
        slicer.prepare_render_z().unwrap();
        let mvp =
            slicer.renderer.projection * slicer.renderer.camera * slicer.renderer.model_matrix;

        // Bottom-left corner of Z-slice (min_x, min_y, 0)
        let p_bl = mvp * glam::vec4(-5.0, -5.0, 0.0, 1.0);
        let ndc_bl = p_bl.xyz() / p_bl.w;
        assert!((ndc_bl.x + 1.0).abs() < 1e-6);
        assert!((ndc_bl.y + 1.0).abs() < 1e-6);

        // Top-right corner of Z-slice (max_x, max_y, 0)
        let p_tr = mvp * glam::vec4(5.0, 5.0, 0.0, 1.0);
        let ndc_tr = p_tr.xyz() / p_tr.w;
        assert!((ndc_tr.x - 1.0).abs() < 1e-6);
        assert!((ndc_tr.y - 1.0).abs() < 1e-6);

        // Test Y-slicing
        slicer.prepare_render_y().unwrap();
        let mvp_y =
            slicer.renderer.projection * slicer.renderer.camera * slicer.renderer.model_matrix;

        // Bottom-left corner of Y-slice (min_x, 0, min_z)
        let p_bl_y = mvp_y * glam::vec4(-5.0, 0.0, -5.0, 1.0);
        let ndc_bl_y = p_bl_y.xyz() / p_bl_y.w;
        assert!((ndc_bl_y.x + 1.0).abs() < 1e-6);
        assert!((ndc_bl_y.y + 1.0).abs() < 1e-6);

        // Top-right corner of Y-slice (max_x, 0, max_z)
        let p_tr_y = mvp_y * glam::vec4(5.0, 0.0, 5.0, 1.0);
        let ndc_tr_y = p_tr_y.xyz() / p_tr_y.w;
        assert!((ndc_tr_y.x - 1.0).abs() < 1e-6);
        assert!((ndc_tr_y.y - 1.0).abs() < 1e-6);

        // Test X-slicing
        slicer.prepare_render_x().unwrap();
        let mvp_x =
            slicer.renderer.projection * slicer.renderer.camera * slicer.renderer.model_matrix;

        // Bottom-left corner of X-slice (0, min_y, min_z)
        let p_bl_x = mvp_x * glam::vec4(0.0, -5.0, -5.0, 1.0);
        let ndc_bl_x = p_bl_x.xyz() / p_bl_x.w;
        assert!((ndc_bl_x.x + 1.0).abs() < 1e-6);
        assert!((ndc_bl_x.y + 1.0).abs() < 1e-6);

        // Top-right corner of X-slice (0, max_y, max_z)
        let p_tr_x = mvp_x * glam::vec4(0.0, 5.0, 5.0, 1.0);
        let ndc_tr_x = p_tr_x.xyz() / p_tr_x.w;
        assert!((ndc_tr_x.x - 1.0).abs() < 1e-6);
        assert!((ndc_tr_x.y - 1.0).abs() < 1e-6);
    }
}
