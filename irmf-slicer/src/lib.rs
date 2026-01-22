pub mod irmf;
pub mod wgpu_renderer;

pub use image::DynamicImage;
pub use irmf::{IrmfError, IrmfHeader, IrmfModel};
pub use wgpu_renderer::WgpuRenderer;

pub type IrmfResult<T> = Result<T, IrmfError>;

pub trait Renderer {
    fn init(&mut self, width: u32, height: u32) -> IrmfResult<()>;
    fn prepare(
        &mut self,
        model: &IrmfModel,
        vertices: &[f32],
        projection: glam::Mat4,
        camera: glam::Mat4,
        model_matrix: glam::Mat4,
        vec3_str: &str,
    ) -> IrmfResult<()>;
    fn render(&mut self, slice_depth: f32, material_num: usize) -> IrmfResult<DynamicImage>;
}

pub struct Slicer<R: Renderer> {
    pub model: IrmfModel,
    pub renderer: R,
    pub res_x: f32, // microns
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

    pub fn num_z_slices(&self) -> usize {
        let delta_z = self.res_z / 1000.0;
        let min_z = self.model.header.min[2];
        let max_z = self.model.header.max[2];
        (0.5 + (max_z - min_z) / delta_z).floor() as usize
    }

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
