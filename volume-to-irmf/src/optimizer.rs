use crate::volume::VoxelVolume;
use anyhow::Result;
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Statistics from the optimization process.
pub struct Stats {
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total duration of the optimization.
    pub duration: std::time::Duration,
    /// Final error value after optimization.
    pub final_error: f32,
}

/// Optimizer for converting voxel volumes to optimized IRMF shaders using constructive solid geometry.
pub struct Optimizer {
    /// The target voxel volume to approximate.
    pub target_volume: Arc<VoxelVolume>,
    /// List of cuboids found during optimization.
    pub cuboids: Vec<Cuboid>,
    /// Low-frequency Fourier coefficients.
    pub fourier_coefficients: Vec<Complex<f32>>,
    /// Results from pass 2 (X-runs).
    pub pass2_results: Vec<[i32; 4]>,
    /// Results from pass 3 (XY-planes).
    pub pass3_results: Vec<([i32; 4], i32)>,
    /// Statistics from the optimization process.
    pub stats: Stats,
    /// Whether to use GPU for optimization.
    pub use_gpu: bool,
    /// WGPU device for GPU computations.
    device: Option<wgpu::Device>,
    /// WGPU queue for submitting commands.
    queue: Option<wgpu::Queue>,
}

/// A rectangular cuboid defined by its bounding coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Cuboid {
    /// Minimum X coordinate.
    pub x1: i32,
    /// Maximum X coordinate.
    pub x2: i32,
    /// Minimum Y coordinate.
    pub y1: i32,
    /// Maximum Y coordinate.
    pub y2: i32,
    /// Minimum Z coordinate.
    pub z1: i32,
    /// Maximum Z coordinate.
    pub z2: i32,
}

impl Optimizer {
    /// Creates a new optimizer for the given voxel volume.
    ///
    /// Initializes the WGPU device and prepares for optimization.
    ///
    /// # Arguments
    ///
    /// * `target_volume` - The voxel volume to optimize.
    /// * `use_gpu` - Whether to use GPU for optimization.
    pub async fn new(target_volume: VoxelVolume, use_gpu: bool) -> Result<Self> {
        let (device, queue) = if use_gpu {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok_or_else(|| anyhow::anyhow!("No WGPU adapter"))?;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("Optimizer Device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits {
                            max_buffer_size: adapter.limits().max_buffer_size,
                            max_storage_buffer_binding_size: adapter
                                .limits()
                                .max_storage_buffer_binding_size,
                            ..wgpu::Limits::default()
                        },
                        memory_hints: Default::default(),
                    },
                    None,
                )
                .await?;

            device.on_uncaptured_error(Box::new(|error| {
                panic!("WGPU error: {}", error);
            }));
            (Some(device), Some(queue))
        } else {
            (None, None)
        };

        Ok(Self {
            target_volume: Arc::new(target_volume),
            cuboids: Vec::new(),
            fourier_coefficients: Vec::new(),
            pass2_results: Vec::new(),
            pass3_results: Vec::new(),
            stats: Stats {
                iterations: 0,
                duration: std::time::Duration::from_secs(0),
                final_error: 0.0,
            },
            use_gpu,
            device,
            queue,
        })
    }

    /// Runs the lossless cuboid merging algorithm to optimize the voxel volume.
    ///
    /// This performs multiple passes to identify and merge adjacent voxels into
    /// larger cuboids for more efficient representation.
    pub async fn run_lossless(&mut self) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Pass 1: CPU - Voxel to Map
        println!("Pass 1: Voxel to Map...");
        let mut yz_to_x = BTreeMap::new();
        let dims = self.target_volume.dims;
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    if self.target_volume.get(x, y, z) > 0.5 {
                        yz_to_x
                            .entry((y as i32, z as i32))
                            .or_insert_with(Vec::new)
                            .push(x as i32);
                    }
                }
            }
        }

        if yz_to_x.is_empty() {
            println!("Volume is empty, skipping further passes.");
            self.stats.duration = start_time.elapsed();
            self.stats.iterations = 1;
            return Ok(());
        }

        // Pass 2: Merge X
        if self.use_gpu {
            println!("Pass 2: GPU Merge X...");
            self.pass2_results = self.run_gpu_pass2(&yz_to_x).await?;
        } else {
            println!("Pass 2: CPU Merge X...");
            self.pass2_results = self.run_cpu_pass2(&yz_to_x);
        }
        println!("Pass 2 produced {} X-runs.", self.pass2_results.len());

        // Preparation for Pass 3: Map Z -> Sorted List of vec4i
        println!("Preparing Pass 3...");
        let mut z_to_runs = BTreeMap::new();
        for run in &self.pass2_results {
            z_to_runs.entry(run[3]).or_insert_with(Vec::new).push(*run);
        }
        for runs in z_to_runs.values_mut() {
            // Sort first by X1, then by Y
            runs.sort_by(|a, b| {
                if a[0] != b[0] {
                    a[0].cmp(&b[0])
                } else {
                    a[2].cmp(&b[2])
                }
            });
        }

        // Pass 3: Merge Y
        if self.use_gpu {
            println!("Pass 3: GPU Merge Y...");
            self.pass3_results = self.run_gpu_pass3(&z_to_runs).await?;
        } else {
            println!("Pass 3: CPU Merge Y...");
            self.pass3_results = self.run_cpu_pass3(&z_to_runs);
        }
        println!("Pass 3 produced {} XY-planes.", self.pass3_results.len());

        // Pass 4: CPU - Merge Z
        println!("Pass 4: CPU Merge Z...");
        self.cuboids = self.run_cpu_pass4();
        println!("Pass 4 produced {} final cuboids.", self.cuboids.len());

        self.stats.duration = start_time.elapsed();
        self.stats.iterations = 1;
        self.stats.final_error = 0.0;

        Ok(())
    }

    /// Runs the Fourier approximation algorithm to generate a continuous scalar field.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of low-frequency coefficients to keep in each dimension.
    pub async fn run_fourier(&mut self, k: usize) -> Result<()> {
        let start_time = std::time::Instant::now();
        println!("Running Fourier approximation (k={})...", k);

        self.fourier_coefficients = self.target_volume.generate_fourier_coefficients(k);

        self.stats.duration = start_time.elapsed();
        self.stats.iterations = 1;
        self.stats.final_error = 0.0; // TODO: compute error

        Ok(())
    }

    /// Generates IRMF shader code using the Fourier approximation.
    pub fn generate_fourier_irmf(&self, language: String) -> String {
        let k = (self.fourier_coefficients.len() as f32)
            .powf(1.0 / 3.0)
            .round() as usize;
        let mut coeffs_re = String::new();
        let mut coeffs_im = String::new();

        for (i, c) in self.fourier_coefficients.iter().enumerate() {
            if i > 0 {
                coeffs_re.push_str(", ");
                coeffs_im.push_str(", ");
                if i % 8 == 0 {
                    coeffs_re.push('\n');
                    coeffs_im.push('\n');
                    coeffs_re.push_str("        ");
                    coeffs_im.push_str("        ");
                }
            }
            coeffs_re.push_str(&format!("{:.6}", c.re));
            coeffs_im.push_str(&format!("{:.6}", c.im));
        }

        let array_decl = if language == "glsl" {
            format!(
                "const float coeffs_re[{}] = float[](\n        {}\n    );\n    const float coeffs_im[{}] = float[](\n        {}\n    );",
                self.fourier_coefficients.len(),
                coeffs_re,
                self.fourier_coefficients.len(),
                coeffs_im
            )
        } else {
            format!(
                "const coeffs_re = array<f32, {}>(\n        {}\n    );\n    const coeffs_im = array<f32, {}>(\n        {}\n    );",
                self.fourier_coefficients.len(),
                coeffs_re,
                self.fourier_coefficients.len(),
                coeffs_im
            )
        };

        let mut reconstruction_logic = String::new();
        if language == "glsl" {
            reconstruction_logic.push_str(&format!(r###"
    float d = 0.0;
    float TWO_PI = 6.28318530718;
    for (int wz = 0; wz < {K}; wz++) {{
        for (int wy = 0; wy < {K}; wy++) {{
            for (int wx = 0; wx < {K}; wx++) {{
                int idx = wz * {K} * {K} + wy * {K} + wx;
                float angle = TWO_PI * (float(wx) * v.x / DIMS.x + float(wy) * v.y / DIMS.y + float(wz) * v.z / DIMS.z);
                d += coeffs_re[idx] * cos(angle) - coeffs_im[idx] * sin(angle);
            }}
        }}
    }}
    if (d < 0.0) {{ materials = solidMaterial; return; }}
"###, K = k));
        } else {
            reconstruction_logic.push_str(&format!(r###"
    var d: f32 = 0.0;
    const TWO_PI: f32 = 6.28318530718;
    for (var wz: i32 = 0; wz < {K}; wz++) {{
        for (var wy: i32 = 0; wy < {K}; wy++) {{
            for (var wx: i32 = 0; wx < {K}; wx++) {{
                let idx = wz * {K} * {K} + wy * {K} + wx;
                let angle = TWO_PI * (f32(wx) * v.x / DIMS.x + f32(wy) * v.y / DIMS.y + f32(wz) * v.z / DIMS.z);
                d += coeffs_re[idx] * cos(angle) - coeffs_im[idx] * sin(angle);
            }}
        }}
    }}
    if (d < 0.0) {{ return solidMaterial; }}
"###, K = k));
        }

        if language == "glsl" {
            self.wrap_glsl_irmf(array_decl, reconstruction_logic, "Fourier Approximation")
        } else {
            self.wrap_wgsl_irmf(array_decl, reconstruction_logic, "Fourier Approximation")
        }
    }

    fn run_cpu_pass2(&self, yz_to_x: &BTreeMap<(i32, i32), Vec<i32>>) -> Vec<[i32; 4]> {
        yz_to_x
            .par_iter()
            .map(|(&(y, z), x_indices)| {
                let mut results = Vec::new();
                if x_indices.is_empty() {
                    return results;
                }

                let mut x_start = x_indices[0];
                let mut x_prev = x_start;

                for &x_curr in &x_indices[1..] {
                    if x_curr != x_prev + 1 {
                        results.push([x_start, x_prev, y, z]);
                        x_start = x_curr;
                    }
                    x_prev = x_curr;
                }
                results.push([x_start, x_prev, y, z]);
                results
            })
            .flatten()
            .collect()
    }

    async fn run_gpu_pass2(
        &self,
        yz_to_x: &BTreeMap<(i32, i32), Vec<i32>>,
    ) -> Result<Vec<[i32; 4]>> {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let shader_src = include_str!("pass2.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pass 2 Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let dims = self.target_volume.dims;
        let num_yz = dims[1] * dims[2];

        let mut x_indices: Vec<i32> = Vec::new();
        let mut yz_offsets = vec![0u32; num_yz as usize];
        let mut yz_counts = vec![0u32; num_yz as usize];

        for z in 0..dims[2] {
            for y in 0..dims[1] {
                let idx = (z * dims[1] + y) as usize;
                if let Some(xs) = yz_to_x.get(&(y as i32, z as i32)) {
                    yz_offsets[idx] = x_indices.len() as u32;
                    yz_counts[idx] = xs.len() as u32;
                    x_indices.extend(xs);
                }
            }
        }

        let x_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("X Indices Buffer"),
            contents: bytemuck::cast_slice(&x_indices),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Offsets Buffer"),
            contents: bytemuck::cast_slice(&yz_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Counts Buffer"),
            contents: bytemuck::cast_slice(&yz_counts),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Max possible results is one per voxel
        let max_results = self.target_volume.data.len();
        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (max_results * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let result_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&result_count_buffer, 0, bytemuck::cast_slice(&[0u32]));

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct Config {
            num_yz: u32,
            dims_y: u32,
            dims_z: u32,
            pad: u32,
        }
        let config = Config {
            num_yz,
            dims_y: dims[1],
            dims_z: dims[2],
            pad: 0,
        };
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::cast_slice(&[config]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pass 2 Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pass 2 Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: result_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
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
            compute_pass.dispatch_workgroups(num_yz.div_ceil(64), 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (max_results * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (max_results * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(&result_count_buffer, 0, &count_staging_buffer, 0, 4);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Out of Memory error in Pass 2: {}", error);
        }
        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Validation error in Pass 2: {}", error);
        }

        let count = {
            let (tx, rx) = futures::channel::oneshot::channel();
            count_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    tx.send(res).unwrap();
                });
            device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let data = count_staging_buffer.slice(..).get_mapped_range();
            let count = bytemuck::cast_slice::<u8, u32>(&data)[0];
            drop(data);
            count_staging_buffer.unmap();
            count
        };

        let results = {
            let (tx, rx) = futures::channel::oneshot::channel();
            staging_buffer.slice(..(count as u64 * 16)).map_async(
                wgpu::MapMode::Read,
                move |res| {
                    tx.send(res).unwrap();
                },
            );
            device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let data = staging_buffer
                .slice(..(count as u64 * 16))
                .get_mapped_range();
            let res: Vec<[i32; 4]> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            res
        };

        Ok(results)
    }

    fn run_cpu_pass3(&self, z_to_runs: &BTreeMap<i32, Vec<[i32; 4]>>) -> Vec<([i32; 4], i32)> {
        z_to_runs
            .par_iter()
            .map(|(&z, runs)| {
                let mut results = Vec::new();
                if runs.is_empty() {
                    return results;
                }

                let first = runs[0];
                let mut x1 = first[0];
                let mut x2 = first[1];
                let mut y_start = first[2];
                let mut y_prev = y_start;

                for &curr in &runs[1..] {
                    if curr[0] == x1 && curr[1] == x2 && curr[2] == y_prev + 1 {
                        y_prev = curr[2];
                    } else {
                        results.push(([x1, x2, y_start, y_prev], z));
                        x1 = curr[0];
                        x2 = curr[1];
                        y_start = curr[2];
                        y_prev = y_start;
                    }
                }
                results.push(([x1, x2, y_start, y_prev], z));
                results
            })
            .flatten()
            .collect()
    }

    async fn run_gpu_pass3(
        &self,
        z_to_runs: &BTreeMap<i32, Vec<[i32; 4]>>,
    ) -> Result<Vec<([i32; 4], i32)>> {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let shader_src = include_str!("pass3.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Pass 3 Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let dims = self.target_volume.dims;
        let num_z = dims[2];

        let mut all_runs: Vec<[i32; 4]> = Vec::new();
        let mut z_offsets = vec![0u32; num_z as usize];
        let mut z_counts = vec![0u32; num_z as usize];

        for z in 0..num_z {
            if let Some(runs) = z_to_runs.get(&(z as i32)) {
                z_offsets[z as usize] = all_runs.len() as u32;
                z_counts[z as usize] = runs.len() as u32;
                all_runs.extend(runs);
            }
        }

        let runs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Runs Buffer"),
            contents: bytemuck::cast_slice(&all_runs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let offset_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Z Offsets Buffer"),
            contents: bytemuck::cast_slice(&z_offsets),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let count_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Z Counts Buffer"),
            contents: bytemuck::cast_slice(&z_counts),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let max_results = all_runs.len();
        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (max_results * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let results_extra_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Extra Buffer"),
            size: (max_results * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let result_count_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&result_count_buffer, 0, bytemuck::cast_slice(&[0u32]));

        #[repr(C)]
        #[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
        struct Config {
            num_z: u32,
        }
        let config = Config { num_z };
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Config Buffer"),
            contents: bytemuck::cast_slice(&[config]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Pass 3 Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pass 3 Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: runs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offset_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: results_extra_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: result_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
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
            compute_pass.dispatch_workgroups(num_z.div_ceil(64), 1, 1);
        }

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (max_results * 16) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let extra_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Extra Staging Buffer"),
            size: (max_results * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let count_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Count Staging Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(
            &results_buffer,
            0,
            &staging_buffer,
            0,
            (max_results * 16) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &results_extra_buffer,
            0,
            &extra_staging_buffer,
            0,
            (max_results * 4) as u64,
        );
        encoder.copy_buffer_to_buffer(&result_count_buffer, 0, &count_staging_buffer, 0, 4);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Out of Memory error in Pass 3: {}", error);
        }
        if let Some(error) = device.pop_error_scope().await {
            anyhow::bail!("WGPU Validation error in Pass 3: {}", error);
        }

        let count = {
            let (tx, rx) = futures::channel::oneshot::channel();
            count_staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    tx.send(res).unwrap();
                });
            device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let data = count_staging_buffer.slice(..).get_mapped_range();
            let count = bytemuck::cast_slice::<u8, u32>(&data)[0];
            drop(data);
            count_staging_buffer.unmap();
            count
        };

        let results = {
            let (tx, rx) = futures::channel::oneshot::channel();
            staging_buffer.slice(..(count as u64 * 16)).map_async(
                wgpu::MapMode::Read,
                move |res| {
                    tx.send(res).unwrap();
                },
            );
            device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let data = staging_buffer
                .slice(..(count as u64 * 16))
                .get_mapped_range();
            let res: Vec<[i32; 4]> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            res
        };

        let extras = {
            let (tx, rx) = futures::channel::oneshot::channel();
            extra_staging_buffer.slice(..(count as u64 * 4)).map_async(
                wgpu::MapMode::Read,
                move |res| {
                    tx.send(res).unwrap();
                },
            );
            device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let data = extra_staging_buffer
                .slice(..(count as u64 * 4))
                .get_mapped_range();
            let res: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            extra_staging_buffer.unmap();
            res
        };

        let combined = results.into_iter().zip(extras.into_iter()).collect();
        Ok(combined)
    }

    fn run_cpu_pass4(&self) -> Vec<Cuboid> {
        // Pass 3 results are tuples (([X1, X2, Y1, Y2]), Z)
        // Preparation for Pass 4: sort by X1, then Y1, then Z
        let mut sorted_pass3 = self.pass3_results.clone();
        sorted_pass3.sort_by(|a, b| {
            if a.0[0] != b.0[0] {
                a.0[0].cmp(&b.0[0])
            } else if a.0[2] != b.0[2] {
                a.0[2].cmp(&b.0[2])
            } else if a.0[1] != b.0[1] {
                a.0[1].cmp(&b.0[1])
            } else if a.0[3] != b.0[3] {
                a.0[3].cmp(&b.0[3])
            } else {
                a.1.cmp(&b.1)
            }
        });

        let mut final_cuboids = Vec::new();
        if sorted_pass3.is_empty() {
            return final_cuboids;
        }

        let mut it = sorted_pass3.into_iter();
        let first = it.next().unwrap();
        let mut current = Cuboid {
            x1: first.0[0],
            x2: first.0[1],
            y1: first.0[2],
            y2: first.0[3],
            z1: first.1,
            z2: first.1,
        };

        for next in it {
            if next.0[0] == current.x1
                && next.0[1] == current.x2
                && next.0[2] == current.y1
                && next.0[3] == current.y2
                && next.1 == current.z2 + 1
            {
                current.z2 = next.1;
            } else {
                final_cuboids.push(current);
                current = Cuboid {
                    x1: next.0[0],
                    x2: next.0[1],
                    y1: next.0[2],
                    y2: next.0[3],
                    z1: next.1,
                    z2: next.1,
                };
            }
        }
        final_cuboids.push(current);

        final_cuboids
    }

    pub fn generate_irmf(&self, language: String) -> String {
        self.generate_final_irmf(language)
    }

    /// Generates IRMF shader code for the intermediate Pass 2 results (X-runs).
    pub fn generate_pass2_irmf(&self) -> String {
        let mut buckets: BTreeMap<i32, Vec<[i32; 4]>> = BTreeMap::new();
        for res in &self.pass2_results {
            buckets.entry(res[3]).or_default().push(*res);
        }

        let mut primitives_code = String::new();
        primitives_code.push_str("    switch (iz) {\n");
        for (z, runs) in buckets {
            primitives_code.push_str(&format!("        case {}: {{\n", z));
            for res in runs {
                primitives_code.push_str(&format!(
                    "            if (cuboid(vi, vec3i({}, {}, {}), vec3i({}, {}, {}))) {{ return solidMaterial; }}\n",
                    res[0], res[2], res[3], res[1], res[2], res[3],
                ));
            }
            primitives_code.push_str("        }\n");
        }
        primitives_code.push_str("        default: {}\n    }\n");
        self.wrap_wgsl_irmf("".to_string(), primitives_code, "Pass 2 (X-runs)")
    }

    /// Generates IRMF shader code for the intermediate Pass 3 results (XY-planes).
    pub fn generate_pass3_irmf(&self) -> String {
        let mut buckets: BTreeMap<i32, Vec<[i32; 4]>> = BTreeMap::new();
        for (rect, z) in &self.pass3_results {
            buckets.entry(*z).or_default().push(*rect);
        }

        let mut primitives_code = String::new();
        primitives_code.push_str("    switch (iz) {\n");
        for (z, rects) in buckets {
            primitives_code.push_str(&format!("        case {}: {{\n", z));
            for r in rects {
                primitives_code.push_str(&format!(
                    "            if (cuboid(vi, vec3i({}, {}, {}), vec3i({}, {}, {}))) {{ return solidMaterial; }}\n",
                    r[0], r[2], z, r[1], r[3], z,
                ));
            }
            primitives_code.push_str("        }\n");
        }
        primitives_code.push_str("        default: {}\n    }\n");
        self.wrap_wgsl_irmf("".to_string(), primitives_code, "Pass 3 (XY-planes)")
    }

    /// Generates the final optimized IRMF shader code.
    ///
    /// # Arguments
    ///
    /// * `language` - The shader language ("glsl" or "wgsl").
    pub fn generate_final_irmf(&self, language: String) -> String {
        let bucket_size_z = (self.target_volume.dims[2] as f32 / 32.0).ceil() as i32;
        let bucket_size_y = (self.target_volume.dims[1] as f32 / 32.0).ceil() as i32;
        let bucket_size_z = bucket_size_z.max(1);
        let bucket_size_y = bucket_size_y.max(1);

        let mut buckets: BTreeMap<i32, BTreeMap<i32, std::collections::BTreeSet<usize>>> =
            BTreeMap::new();
        for (i, c) in self.cuboids.iter().enumerate() {
            let bz1 = c.z1 / bucket_size_z;
            let bz2 = c.z2 / bucket_size_z;
            let by1 = c.y1 / bucket_size_y;
            let by2 = c.y2 / bucket_size_y;
            for bz in bz1..=bz2 {
                let y_buckets = buckets.entry(bz).or_default();
                for by in by1..=by2 {
                    y_buckets.entry(by).or_default().insert(i);
                }
            }
        }

        let mut bz_by_cases = String::new();
        let mut bz_cases = String::new();
        let mut primitives_code = String::new();
        let int_let = if language == "glsl" { "int" } else { "let" };
        let vec3i = if language == "glsl" { "ivec3" } else { "vec3i" };
        primitives_code.push_str(&format!("    {} bz = vi.z / {};\n", int_let, bucket_size_z));
        primitives_code.push_str(&format!("    {} by = vi.y / {};\n", int_let, bucket_size_y));
        for (bz, y_buckets) in buckets {
            if language == "glsl" {
                primitives_code.push_str(&format!(
                    "    if (bz == {}) {{ materials = bz{}Case(vi, by); return; }}\n",
                    bz, bz
                ));
                bz_cases.push_str(&format!("\nvec4 bz{}Case(ivec3 vi, int by) {{\n", bz));
            } else {
                primitives_code.push_str(&format!(
                    "    if (bz == {}) {{ return bz{}Case(vi, by); }}\n",
                    bz, bz
                ));
                bz_cases.push_str(&format!(
                    "fn bz{}Case(vi: vec3i, by: i32) -> vec4f {{\n",
                    bz
                ));
            }
            for (by, cuboid_indices) in y_buckets {
                bz_cases.push_str(&format!(
                    "    if (by == {}) {{ return bz{}by{}Case(vi); }}\n",
                    by, bz, by
                ));
                if language == "glsl" {
                    bz_by_cases.push_str(&format!("vec4 bz{}by{}Case(ivec3 vi) {{\n", bz, by));
                } else {
                    bz_by_cases
                        .push_str(&format!("fn bz{}by{}Case(vi: vec3i) -> vec4f {{\n", bz, by));
                }
                for i in cuboid_indices {
                    let c = &self.cuboids[i];
                    bz_by_cases.push_str(&format!(
                        "    if (cuboid(vi, {}({}, {}, {}), {}({}, {}, {}))) {{ return solidMaterial; }}\n",
                        vec3i, c.x1, c.y1, c.z1, vec3i, c.x2, c.y2, c.z2,
                    ));
                }
                if language == "glsl" {
                    bz_by_cases.push_str("    return vec4(0,0,0,0);\n}\n\n");
                } else {
                    bz_by_cases.push_str("    return vec4f(0,0,0,0);\n}\n\n");
                }
            }
            if language == "glsl" {
                bz_cases.push_str("    return vec4(0,0,0,0);\n");
            } else {
                bz_cases.push_str("    return vec4f(0,0,0,0);\n");
            }
            bz_cases.push_str("}\n");
        }
        bz_by_cases.push_str(&bz_cases);
        if language == "glsl" {
            return self.wrap_glsl_irmf(bz_by_cases, primitives_code, "Final Lossless Cuboids");
        }
        self.wrap_wgsl_irmf(bz_by_cases, primitives_code, "Final Lossless Cuboids")
    }

    pub fn cuboid_count(&self) -> usize {
        self.cuboids.len()
    }

    fn wrap_glsl_irmf(
        &self,
        helper_functions: String,
        primitives_code: String,
        notes: &str,
    ) -> String {
        let min = self.target_volume.min;
        let max = self.target_volume.max;
        let dims = self.target_volume.dims;

        format!(
            r###"/*{{
  "irmf": "1.0",
  "language": "glsl",
  "materials": ["Material"],
  "min": [{:.4}, {:.4}, {:.4}],
  "max": [{:.4}, {:.4}, {:.4}],
  "notes": "{}",
  "units": "mm"
}}*/

const vec3 MIN_BOUND = vec3({:.4}, {:.4}, {:.4});
const vec3 MAX_BOUND = vec3({:.4}, {:.4}, {:.4});
const vec3 DIMS = vec3({:.1}, {:.1}, {:.1});
const vec3 VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const vec4 solidMaterial = vec4(1.0, 0.0, 0.0, 0.0);

bool cuboid(ivec3 v, ivec3 b_min, ivec3 b_max) {{
    return all(greaterThanEqual(v, b_min)) && all(lessThanEqual(v, b_max));
}}

{}
void mainModel4(out vec4 materials, in vec3 xyz) {{
    vec3 v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    ivec3 vi = ivec3(floor(v));
{}
    materials = vec4(0,0,0,0);
}}
"###,
            min.x,
            min.y,
            min.z,
            max.x,
            max.y,
            max.z,
            notes,
            min.x,
            min.y,
            min.z,
            max.x,
            max.y,
            max.z,
            dims[0] as f32,
            dims[1] as f32,
            dims[2] as f32,
            helper_functions,
            primitives_code
        )
    }

    fn wrap_wgsl_irmf(
        &self,
        helper_functions: String,
        primitives_code: String,
        notes: &str,
    ) -> String {
        let min = self.target_volume.min;
        let max = self.target_volume.max;
        let dims = self.target_volume.dims;

        format!(
            r###"/*{{
  "irmf": "1.0",
  "language": "wgsl",
  "materials": ["Material"],
  "min": [{:.4}, {:.4}, {:.4}],
  "max": [{:.4}, {:.4}, {:.4}],
  "notes": "{}",
  "units": "mm"
}}*/

const MIN_BOUND = vec3f({:.4}, {:.4}, {:.4});
const MAX_BOUND = vec3f({:.4}, {:.4}, {:.4});
const DIMS = vec3f({:.1}, {:.1}, {:.1});
const VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const solidMaterial = vec4f(1.0, 0.0, 0.0, 0.0);

fn cuboid(v: vec3i, b_min: vec3i, b_max: vec3i) -> bool {{
    return all(v >= b_min) && all(v <= b_max);
}}

{}
fn mainModel4(xyz: vec3f) -> vec4f {{
    let v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    let vi = vec3i(floor(v + vec3f(0.5)));
    let iy = i32(floor(v.y));
    let iz = i32(floor(v.z));
{}
    return vec4f(0.0, 0.0, 0.0, 0.0);
}}
"###,
            min.x,
            min.y,
            min.z,
            max.x,
            max.y,
            max.z,
            notes,
            min.x,
            min.y,
            min.z,
            max.x,
            max.y,
            max.z,
            dims[0] as f32,
            dims[1] as f32,
            dims[2] as f32,
            helper_functions,
            primitives_code
        )
    }
}
