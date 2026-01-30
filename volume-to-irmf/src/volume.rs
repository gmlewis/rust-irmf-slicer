use glam::Vec3;
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::FftPlanner;
use wgpu::util::DeviceExt;

/// A 3D volume represented as a grid of voxels.
///
/// Each voxel contains a floating-point value typically between 0.0 and 1.0,
/// representing density or occupancy.
pub struct VoxelVolume {
    /// Voxel data as a flat vector, where each element is a density value (0.0 to 1.0).
    pub data: Vec<f32>,
    /// Dimensions of the volume in voxels [width, height, depth].
    pub dims: [u32; 3],
    /// Minimum coordinates of the volume in world space.
    pub min: Vec3,
    /// Maximum coordinates of the volume in world space.
    pub max: Vec3,
}

impl VoxelVolume {
    /// Creates a new empty voxel volume with the given dimensions and bounds.
    ///
    /// # Arguments
    ///
    /// * `dims` - The dimensions [width, height, depth] in voxels.
    /// * `min` - The minimum coordinates in world space.
    /// * `max` - The maximum coordinates in world space.
    pub fn new(dims: [u32; 3], min: Vec3, max: Vec3) -> Self {
        let size = (dims[0] * dims[1] * dims[2]) as usize;
        Self {
            data: vec![0.0; size],
            dims,
            min,
            max,
        }
    }

    /// Gets the density value at the specified voxel coordinates.
    ///
    /// Returns 0.0 if the coordinates are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (0-based).
    /// * `y` - Y coordinate (0-based).
    /// * `z` - Z coordinate (0-based).
    pub fn get(&self, x: u32, y: u32, z: u32) -> f32 {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return 0.0;
        }
        let idx = (z * self.dims[1] * self.dims[0] + y * self.dims[0] + x) as usize;
        self.data[idx]
    }

    /// Sets the density value at the specified voxel coordinates.
    ///
    /// Does nothing if the coordinates are out of bounds.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate (0-based).
    /// * `y` - Y coordinate (0-based).
    /// * `z` - Z coordinate (0-based).
    /// * `val` - The density value to set (typically 0.0 to 1.0).
    pub fn set(&mut self, x: u32, y: u32, z: u32, val: f32) {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return;
        }
        let idx = (z * self.dims[1] * self.dims[0] + y * self.dims[0] + x) as usize;
        self.data[idx] = val;
    }

    /// Loads a voxel volume from a BinVox file format.
    ///
    /// BinVox is a simple 3D voxel data format. This method parses the file
    /// and creates a `VoxelVolume` with the appropriate dimensions and data.
    ///
    /// # Arguments
    ///
    /// * `reader` - A reader that provides the BinVox file data.
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
            if parts.is_empty() {
                break;
            }
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

        // Binvox scale often refers to the Z axis count for non-uniform models.
        // We calculate factor based on the dimension that matches the scale in world units.
        // For the Rodin coil, scale 100 matches dims[2]=200 (0.5mm/voxel).
        let factor = scale / dims[2] as f32;
        let world_dims = Vec3::new(dims[0] as f32, dims[1] as f32, dims[2] as f32) * factor;
        let mut volume = Self::new(dims, translate, translate + world_dims);
        println!(
            "Binvox header: dims={:?}, translate={:?}, scale={}, world_dims={:?}",
            dims, translate, scale, world_dims
        );

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
                    // Binvox (x, z, y) where x is fastest, z middle, y slowest.
                    let x = voxels_read as u32 % dims[0];
                    let z = (voxels_read as u32 / dims[0]) % dims[2];
                    let y = voxels_read as u32 / (dims[0] * dims[2]);

                    // Map world-space (x, y, z) to our volume indices.
                    // volume.set(x, y, z) uses logical (X, Y, Z) order.
                    volume.set(x, y, z, val_f);
                    voxels_read += 1;
                }
            }
        }

        Ok(volume.tighten())
    }

    pub fn tighten(self) -> Self {
        let mut min_v = [u32::MAX; 3];
        let mut max_v = [0u32; 3];
        let mut any_filled = false;

        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    if self.get(x, y, z) > 0.5 {
                        any_filled = true;
                        min_v[0] = min_v[0].min(x);
                        min_v[1] = min_v[1].min(y);
                        min_v[2] = min_v[2].min(z);
                        max_v[0] = max_v[0].max(x);
                        max_v[1] = max_v[1].max(y);
                        max_v[2] = max_v[2].max(z);
                    }
                }
            }
        }

        if !any_filled {
            return self;
        }

        let new_dims = [
            max_v[0] - min_v[0] + 1,
            max_v[1] - min_v[1] + 1,
            max_v[2] - min_v[2] + 1,
        ];

        let factor_x = (self.max.x - self.min.x) / self.dims[0] as f32;
        let factor_y = (self.max.y - self.min.y) / self.dims[1] as f32;
        let factor_z = (self.max.z - self.min.z) / self.dims[2] as f32;

        let new_min = Vec3::new(
            self.min.x + min_v[0] as f32 * factor_x,
            self.min.y + min_v[1] as f32 * factor_y,
            self.min.z + min_v[2] as f32 * factor_z,
        );
        let new_max = Vec3::new(
            self.min.x + (max_v[0] as f32 + 1.0) * factor_x,
            self.min.y + (max_v[1] as f32 + 1.0) * factor_y,
            self.min.z + (max_v[2] as f32 + 1.0) * factor_z,
        );

        let mut new_volume = Self::new(new_dims, new_min, new_max);
        for z in 0..new_dims[2] {
            for y in 0..new_dims[1] {
                for x in 0..new_dims[0] {
                    new_volume.set(x, y, z, self.get(x + min_v[0], y + min_v[1], z + min_v[2]));
                }
            }
        }
        println!("Tightened volume from {:?} to {:?}", self.dims, new_dims);
        println!("Tightened bounds: min={:?}, max={:?}", new_min, new_max);
        new_volume
    }

    pub fn from_stl(
        reader: &mut (impl std::io::Read + std::io::Seek),
        dims: [u32; 3],
    ) -> anyhow::Result<Self> {
        let mesh =
            stl_io::read_stl(reader).map_err(|e| anyhow::anyhow!("STL read error: {:?}", e))?;

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

        let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];

        data.par_chunks_exact_mut((dims[1] * dims[0]) as usize)
            .enumerate()
            .for_each(|(vz, slice)| {
                let z_val = min.z + (vz as f32 + 0.5) / dims[2] as f32 * (max.z - min.z);
                for vx in 0..dims[0] {
                    for vy in 0..dims[1] {
                        let px = min.x + (vx as f32 + 0.5) / dims[0] as f32 * (max.x - min.x);
                        let py = min.y + (vy as f32 + 0.5) / dims[1] as f32 * (max.y - min.y);

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
                                let z_end = intersections[i + 1];

                                if z_val >= z_start && z_val <= z_end {
                                    slice[(vy * dims[0] + vx) as usize] = 1.0;
                                    break;
                                }
                            }
                        }
                    }
                }
            });

        Ok(Self {
            data,
            dims,
            min,
            max,
        })
    }

    pub fn from_slices(
        slices: Vec<image::DynamicImage>,
        min: Vec3,
        max: Vec3,
    ) -> anyhow::Result<Self> {
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

    pub fn cpu_voxelize(
        vertices: Vec<Vec3>,
        indices: Vec<u32>,
        dims: [u32; 3],
        min: Vec3,
        max: Vec3,
    ) -> anyhow::Result<Self> {
        let num_triangles = indices.len() / 3;
        let world_size = max - min;
        let voxel_size = world_size / Vec3::new(dims[0] as f32, dims[1] as f32, dims[2] as f32);

        let mut data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];

        data.par_chunks_exact_mut((dims[1] * dims[0]) as usize)
            .enumerate()
            .for_each(|(_vz, _slice)| {
                // Actually, the GPU version works by X and Y columns.
            });

        // Redoing to match GPU logic: parallelize over X and Y
        let mut columns: Vec<Vec<f32>> = (0..dims[0] * dims[1])
            .into_par_iter()
            .map(|idx| {
                let x = idx % dims[0];
                let y = idx / dims[0];

                let mut bits = vec![0u32; (dims[2] as usize).div_ceil(32)];

                let jitter_x = 0.5123 + 0.0001 * (x % 13) as f32;
                let jitter_y = 0.4789 + 0.0001 * (y % 17) as f32;
                let orig = Vec3::new(
                    min.x + (x as f32 + jitter_x) * voxel_size.x,
                    min.y + (y as f32 + jitter_y) * voxel_size.y,
                    min.z - 0.1 * world_size.z,
                );
                let ray_dir = Vec3::new(0.0, 0.0, 1.0);

                for i in 0..num_triangles {
                    let v0 = vertices[indices[i * 3] as usize];
                    let v1 = vertices[indices[i * 3 + 1] as usize];
                    let v2 = vertices[indices[i * 3 + 2] as usize];

                    let t = intersect_ray_tri(orig, ray_dir, v0, v1, v2);
                    if t >= 0.0 {
                        let world_z = orig.z + t;
                        let z_world_rel = (world_z - min.z) / voxel_size.z;
                        let z_idx = z_world_rel.round() as i32;
                        if z_idx >= 0 && (z_idx as u32) < dims[2] {
                            let z_idx = z_idx as u32;
                            bits[(z_idx / 32) as usize] ^= 1 << (z_idx % 32);
                        }
                    }
                }

                let mut col = vec![0.0f32; dims[2] as usize];
                let mut inside = false;
                for z in 0..dims[2] {
                    if ((bits[(z / 32) as usize] >> (z % 32)) & 1) == 1 {
                        inside = !inside;
                    }
                    col[z as usize] = if inside { 1.0 } else { 0.0 };
                }
                col
            })
            .collect();

        // Reassemble columns into flat data
        let mut final_data = vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize];
        for (idx, col) in columns.drain(..).enumerate() {
            let x = idx as u32 % dims[0];
            let y = idx as u32 / dims[0];
            for (z, val) in col.into_iter().enumerate() {
                let flat_idx = (z as u32 * dims[1] * dims[0] + y * dims[0] + x) as usize;
                final_data[flat_idx] = val;
            }
        }

        Ok(Self {
            data: final_data,
            dims,
            min,
            max,
        })
    }

    pub async fn gpu_voxelize(
        vertices: Vec<Vec3>,
        indices: Vec<u32>,
        dims: [u32; 3],
        min: Vec3,
        max: Vec3,
    ) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("No WGPU adapter"))?;
        let limits = wgpu::Limits {
            max_buffer_size: adapter.limits().max_buffer_size,
            max_storage_buffer_binding_size: adapter.limits().max_storage_buffer_binding_size,
            max_uniform_buffer_binding_size: adapter.limits().max_uniform_buffer_binding_size,
            max_texture_dimension_3d: adapter.limits().max_texture_dimension_3d,
            max_storage_textures_per_shader_stage: adapter
                .limits()
                .max_storage_textures_per_shader_stage,
            ..wgpu::Limits::default()
        };
        println!("Requested limits: {:?}", limits);
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Voxelizer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        device.on_uncaptured_error(Box::new(|error| {
            panic!("WGPU error in gpu_voxelize: {}", error);
        }));

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let shader_src = include_str!("voxelizer.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Voxelizer Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let texture_size = wgpu::Extent3d {
            width: dims[0],
            height: dims[1],
            depth_or_array_layers: dims[2],
        };

        let voxel_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let vertices_f4: Vec<[f32; 4]> = vertices.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices_f4),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct VoxelConfig {
            num_triangles: u32,
            _pad: [u32; 3],
            min_bound: [f32; 4],
            max_bound: [f32; 4],
        }

        let config = VoxelConfig {
            num_triangles: (indices.len() / 3) as u32,
            _pad: [0; 3],
            min_bound: [min.x, min.y, min.z, 0.0],
            max_bound: [max.x, max.y, max.z, 0.0],
        };

        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::cast_slice(&[config]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Voxelizer Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Voxelizer Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &voxel_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let wg_x = dims[0].div_ceil(16);
            let wg_y = dims[1].div_ceil(16);
            println!("Dispatching workgroups: {}x{}x1", wg_x, wg_y);
            compute_pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let unaligned_bytes_per_row = dims[0] * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padding = (align - unaligned_bytes_per_row % align) % align;
        let aligned_bytes_per_row = unaligned_bytes_per_row + padding;
        println!(
            "unaligned_bytes_per_row: {}, aligned_bytes_per_row: {}",
            unaligned_bytes_per_row, aligned_bytes_per_row
        );

        let output_buffer_size = (aligned_bytes_per_row * dims[1] * dims[2]) as u64;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &voxel_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(dims[1]),
                },
            },
            texture_size,
        );

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Out of Memory error in gpu_voxelize: {}", error);
        }
        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Validation error in gpu_voxelize: {}", error);
        }

        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) =
            futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });
        device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            let mut result = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
            let mut non_zero_count = 0;
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    let start =
                        (z * dims[1] * aligned_bytes_per_row + y * aligned_bytes_per_row) as usize;
                    let end = start + unaligned_bytes_per_row as usize;
                    let row: &[f32] = bytemuck::cast_slice(&data[start..end]);
                    for &val in row {
                        if val > 0.0 {
                            non_zero_count += 1;
                        }
                    }
                    result.extend_from_slice(row);
                }
            }
            drop(data);
            println!("Read back {} non-zero voxels from GPU.", non_zero_count);
            Ok(Self {
                data: result,
                dims,
                min,
                max,
            })
        } else {
            anyhow::bail!("GPU readback failed")
        }
    }

    /// Generates a Signed Distance Field (SDF) from the voxel volume.
    ///
    /// Inside voxels (density > 0.5) will have negative distance to the nearest boundary.
    /// Outside voxels (density <= 0.5) will have positive distance to the nearest boundary.
    pub fn generate_sdf(&self) -> Vec<f32> {
        let size = (self.dims[0] * self.dims[1] * self.dims[2]) as usize;

        // 1. Distance to outside (for points inside)
        let mut dist_to_outside = vec![f32::MAX; size];
        let mut any_outside = false;
        for (i, dist) in dist_to_outside.iter_mut().enumerate() {
            if self.data[i] <= 0.5 {
                *dist = 0.0;
                any_outside = true;
            }
        }
        if any_outside {
            self.compute_edt_3d(&mut dist_to_outside);
        }

        // 2. Distance to inside (for points outside)
        let mut dist_to_inside = vec![f32::MAX; size];
        let mut any_inside = false;
        for (i, dist) in dist_to_inside.iter_mut().enumerate() {
            if self.data[i] > 0.5 {
                *dist = 0.0;
                any_inside = true;
            }
        }
        if any_inside {
            self.compute_edt_3d(&mut dist_to_inside);
        }

        // 3. Combine into signed distance
        let mut sdf = vec![0.0f32; size];
        for i in 0..size {
            if self.data[i] > 0.5 {
                // Inside: negative distance to nearest outside
                sdf[i] = -dist_to_outside[i].sqrt();
            } else {
                // Outside: positive distance to nearest inside
                sdf[i] = dist_to_inside[i].sqrt();
            }
        }
        sdf
    }

    fn compute_edt_3d(&self, data: &mut [f32]) {
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        let nz = self.dims[2] as usize;

        // Pass 1: X (independent rows)
        data.par_chunks_exact_mut(nx).for_each(|row| {
            let mut out = vec![0.0; nx];
            Self::dt1d(row, &mut out);
            row.copy_from_slice(&out);
        });

        // Pass 2: Y (independent columns in each X-Z plane)
        data.par_chunks_exact_mut(ny * nx).for_each(|plane| {
            let mut col = vec![0.0; ny];
            let mut out = vec![0.0; ny];
            for x in 0..nx {
                for y in 0..ny {
                    col[y] = plane[y * nx + x];
                }
                Self::dt1d(&col, &mut out);
                for y in 0..ny {
                    plane[y * nx + x] = out[y];
                }
            }
        });

        // Pass 3: Z (independent columns in each X-Y plane)
        let nx_ny = nx * ny;
        let data_ptr = data.as_mut_ptr() as usize;
        (0..nx_ny).into_par_iter().for_each(|idx| {
            let x = idx % nx;
            let y = idx / nx;
            let mut col = vec![0.0; nz];
            let mut out = vec![0.0; nz];
            for (z, item) in col.iter_mut().enumerate().take(nz) {
                let offset = z * nx_ny + y * nx + x;
                *item = unsafe { *(data_ptr as *const f32).add(offset) };
            }
            Self::dt1d(&col, &mut out);
            for (z, &item) in out.iter().enumerate().take(nz) {
                let offset = z * nx_ny + y * nx + x;
                unsafe { *(data_ptr as *mut f32).add(offset) = item };
            }
        });
    }

    /// 1D Distance Transform using the Felzenszwalb-Huttenlocher algorithm.
    /// f is the input array of squared distances, d is the output array.
    fn dt1d(f: &[f32], d: &mut [f32]) {
        let n = f.len();
        if n == 0 {
            return;
        }

        let mut v = vec![0usize; n];
        let mut z = vec![0.0f32; n + 1];
        let mut k = 0usize;
        v[0] = 0;
        z[0] = f32::NEG_INFINITY;
        z[1] = f32::INFINITY;

        for q in 1..n {
            if f[q] == f32::MAX {
                continue;
            }
            // If f[v[k]] is MAX, we just replace it with current q
            if f[v[k]] == f32::MAX {
                v[k] = q;
                continue;
            }

            let mut s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                / (2.0 * (q as f32 - v[k] as f32));
            while s <= z[k] {
                if k == 0 {
                    break;
                }
                k -= 1;
                s = ((f[q] + (q * q) as f32) - (f[v[k]] + (v[k] * v[k]) as f32))
                    / (2.0 * (q as f32 - v[k] as f32));
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
        }

        k = 0;
        for (q, item) in d.iter_mut().enumerate().take(n) {
            while z[k + 1] < q as f32 {
                k += 1;
            }
            if f[v[k]] == f32::MAX {
                *item = f32::MAX;
            } else {
                let dx = q as f32 - v[k] as f32;
                *item = dx * dx + f[v[k]];
            }
        }
    }

    /// Generates low-frequency Fourier coefficients for the volume's SDF.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of coefficients to keep in each dimension (k x k x k).
    pub fn generate_fourier_coefficients(&self, k: usize) -> Vec<Complex<f32>> {
        let sdf = self.generate_sdf();
        let mut complex_data: Vec<Complex<f32>> =
            sdf.iter().map(|&d| Complex::new(d, 0.0)).collect();

        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        let nz = self.dims[2] as usize;

        // Perform 3D FFT
        self.fft_3d(&mut complex_data);

        // Normalize
        let norm = 1.0 / (nx * ny * nz) as f32;
        for val in &mut complex_data {
            *val *= norm;
        }

        // Extract low-frequency coefficients (k x k x k) centered around zero
        let mut result = Vec::with_capacity(k * k * k);
        let half_k = (k / 2) as i32;

        for dz in -half_k..k as i32 - half_k {
            let z = if dz < 0 { nz as i32 + dz } else { dz } as usize;
            for dy in -half_k..k as i32 - half_k {
                let y = if dy < 0 { ny as i32 + dy } else { dy } as usize;
                for dx in -half_k..k as i32 - half_k {
                    let x = if dx < 0 { nx as i32 + dx } else { dx } as usize;
                    let idx = z * ny * nx + y * nx + x;
                    result.push(complex_data[idx]);
                }
            }
        }
        result
    }

    fn fft_3d(&self, data: &mut [Complex<f32>]) {
        let nx = self.dims[0] as usize;
        let ny = self.dims[1] as usize;
        let nz = self.dims[2] as usize;

        let mut planner = FftPlanner::new();

        // Pass 1: X (independent rows)
        let fft_x = planner.plan_fft_forward(nx);
        data.par_chunks_exact_mut(nx).for_each(|row| {
            fft_x.process(row);
        });

        // Pass 2: Y (independent columns in each X-Z plane)
        let fft_y = planner.plan_fft_forward(ny);
        data.par_chunks_exact_mut(ny * nx).for_each(|plane| {
            let mut col = vec![Complex::new(0.0, 0.0); ny];
            for x in 0..nx {
                for y in 0..ny {
                    col[y] = plane[y * nx + x];
                }
                fft_y.process(&mut col);
                for y in 0..ny {
                    plane[y * nx + x] = col[y];
                }
            }
        });

        // Pass 3: Z (independent columns in each X-Y plane)
        let fft_z = planner.plan_fft_forward(nz);
        let nx_ny = nx * ny;
        let data_ptr = data.as_mut_ptr() as usize;
        (0..nx_ny).into_par_iter().for_each(|idx| {
            let x = idx % nx;
            let y = idx / nx;
            let mut col = vec![Complex::new(0.0, 0.0); nz];
            for (z, item) in col.iter_mut().enumerate().take(nz) {
                let offset = z * nx_ny + y * nx + x;
                *item = unsafe { *(data_ptr as *const Complex<f32>).add(offset) };
            }
            fft_z.process(&mut col);
            for (z, &item) in col.iter().enumerate().take(nz) {
                let offset = z * nx_ny + y * nx + x;
                unsafe { *(data_ptr as *mut Complex<f32>).add(offset) = item };
            }
        });
    }
}

fn intersect_tri_xy(px: f32, py: f32, v: &[stl_io::Vector<f32>; 3]) -> Option<f32> {
    let v0 = Vec3::new(v[0][0], v[0][1], v[0][2]);
    let v1 = Vec3::new(v[1][0], v[1][1], v[1][2]);
    let v2 = Vec3::new(v[2][0], v[2][1], v[2][2]);

    let area = 0.5 * (-v1.y * v2.x + v0.y * (-v1.x + v2.x) + v0.x * (v1.y - v2.y) + v1.x * v2.y);
    if area.abs() < 1e-9 {
        return None;
    }

    let s =
        1.0 / (2.0 * area) * (v0.y * v2.x - v0.x * v2.y + (v2.y - v0.y) * px + (v0.x - v2.x) * py);
    let t =
        1.0 / (2.0 * area) * (v0.x * v1.y - v0.y * v1.x + (v0.y - v1.y) * px + (v1.x - v0.x) * py);

    if s >= 0.0 && t >= 0.0 && (1.0 - s - t) >= 0.0 {
        Some(v0.z + s * (v1.z - v0.z) + t * (v2.z - v0.z))
    } else {
        None
    }
}

// Basic ray-triangle intersection (Möller-Trumbore)
fn intersect_ray_tri(orig: Vec3, dir: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = dir.cross(edge2);
    let a = edge1.dot(h);
    if a > -0.000001 && a < 0.000001 {
        return -1.0;
    }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * s.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return -1.0;
    }
    let q = s.cross(edge1);
    let v = f * dir.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return -1.0;
    }
    f * edge2.dot(q)
}
