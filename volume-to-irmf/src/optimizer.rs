use crate::primitives::{BooleanOp, Primitive};
use crate::volume::VoxelVolume;
use anyhow::Result;
use glam::Vec3;
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ErrorResult {
    mse_sum: f32,
    iou_min_sum: f32,
    iou_max_sum: f32,
    sample_idx: u32,
}

pub struct Stats {
    pub iterations: usize,
    pub duration: std::time::Duration,
    pub final_error: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    num_samples: u32,
    num_primitives: u32,
    seed: u32,
    num_candidates: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Perturbation {
    prim_idx: u32,
    pos_delta: Vec3,
    size_scale: f32,
    op: u32,
}

pub struct Optimizer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    target_volume: Arc<VoxelVolume>,
    primitives: Vec<Primitive>,

    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    target_texture_view: wgpu::TextureView,

    config_buffer: wgpu::Buffer,
    primitive_buffer: wgpu::Buffer,
    perturbation_buffer: wgpu::Buffer,
    results_buffer: wgpu::Buffer,
    results_staging_buffer: wgpu::Buffer,
    samples_buffer: wgpu::Buffer,

    num_candidates: u32,
    samples_per_candidate: u32,
    seed: u32,

    samples: Vec<Vec3>,
    last_results: Vec<ErrorResult>,

    pub stats: Stats,
    start_time: std::time::Instant,
}

impl Optimizer {
    pub async fn new(target_volume: VoxelVolume) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or_else(|| anyhow::anyhow!("No suitable WGPU adapter found"))?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Optimizer Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: adapter
                            .limits()
                            .max_storage_buffer_binding_size,
                        max_buffer_size: adapter.limits().max_buffer_size,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let texture_size = wgpu::Extent3d {
            width: target_volume.dims[0],
            height: target_volume.dims[1],
            depth_or_array_layers: target_volume.dims[2],
        };

        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Target Volume Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&target_volume.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * target_volume.dims[0]),
                rows_per_image: Some(target_volume.dims[1]),
            },
            texture_size,
        );

        let target_texture_view =
            target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let shader_src = include_str!("optimizer.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Optimizer Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Optimizer Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Optimizer Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Optimizer Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let num_candidates = 256;
        let samples_per_candidate = 1024 * 32; // 32768
        let total_workgroups = num_candidates * (samples_per_candidate / 256);

        let samples_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Samples Buffer"),
            size: (std::mem::size_of::<Vec3>() * samples_per_candidate as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(samples_per_candidate as usize);
        for _ in 0..samples_per_candidate {
            samples.push(Vec3::new(rng.r#gen(), rng.r#gen(), rng.r#gen()));
        }
        queue.write_buffer(&samples_buffer, 0, bytemuck::cast_slice(&samples));

        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Buffer"),
            size: std::mem::size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitive Buffer"),
            size: (std::mem::size_of::<Primitive>() * 1024) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let perturbation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Perturbation Buffer"),
            size: (std::mem::size_of::<Perturbation>() * num_candidates as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (std::mem::size_of::<ErrorResult>() * total_workgroups as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let results_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Staging Buffer"),
            size: (std::mem::size_of::<ErrorResult>() * total_workgroups as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut optimizer = Self {
            device,
            queue,
            target_volume: Arc::new(target_volume),
            primitives: Vec::new(),
            pipeline,
            bind_group_layout,
            target_texture_view,
            config_buffer,
            primitive_buffer,
            perturbation_buffer,
            results_buffer,
            results_staging_buffer,
            samples_buffer,
            num_candidates,
            samples_per_candidate,
            seed: 0,
            samples,
            last_results: Vec::new(),
            stats: Stats {
                iterations: 0,
                duration: std::time::Duration::from_secs(0),
                final_error: 1.0,
            },
            start_time: std::time::Instant::now(),
        };

        let perts = vec![
            Perturbation {
                prim_idx: 8888,
                pos_delta: Vec3::ZERO,
                size_scale: 1.0,
                op: 0,
            };
            num_candidates as usize
        ];
        optimizer.calculate_errors(&perts).await?;

        Ok(optimizer)
    }

    pub fn add_primitive(&mut self, prim: Primitive) {
        self.primitives.push(prim);
    }

    pub async fn run_iteration(&mut self) -> Result<f32> {
        self.stats.iterations += 1;
        self.stats.duration = self.start_time.elapsed();

        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut seed_pos = Vec3::new(rng.r#gen(), rng.r#gen(), rng.r#gen());
        if !self.last_results.is_empty() {
            let groups_per_cand = self.samples_per_candidate / 256;
            let mut max_err = -1.0;
            let mut worst_group = 0;
            for g in 0..groups_per_cand as usize {
                let res = &self.last_results[g];
                if res.mse_sum > max_err {
                    max_err = res.mse_sum;
                    worst_group = g;
                }
            }
            let sample_idx = worst_group as u32 * 256 + rng.gen_range(0..256);
            seed_pos = self.samples[sample_idx as usize];
        }

        let mut perts = Vec::with_capacity(self.num_candidates as usize);
        perts.push(Perturbation {
            prim_idx: 8888,
            pos_delta: Vec3::ZERO,
            size_scale: 1.0,
            op: 0,
        });

        for i in 1..self.num_candidates / 4 {
            perts.push(Perturbation {
                prim_idx: 9999,
                pos_delta: seed_pos,
                size_scale: 0.01 * i as f32,
                op: BooleanOp::Union as u32,
            });
        }
        for i in self.num_candidates / 4..self.num_candidates / 2 {
            perts.push(Perturbation {
                prim_idx: 9999,
                pos_delta: seed_pos,
                size_scale: 0.01 * (i - self.num_candidates / 4 + 1) as f32,
                op: BooleanOp::Difference as u32,
            });
        }
        for _ in self.num_candidates / 2..self.num_candidates {
            if !self.primitives.is_empty() {
                let prim_idx = rng.gen_range(0..self.primitives.len()) as u32;
                perts.push(Perturbation {
                    prim_idx,
                    pos_delta: Vec3::new(
                        rng.gen_range(-0.02..0.02),
                        rng.gen_range(-0.02..0.02),
                        rng.gen_range(-0.02..0.02),
                    ),
                    size_scale: rng.gen_range(0.95..1.05),
                    op: self.primitives[prim_idx as usize].op,
                });
            } else {
                perts.push(perts[0]);
            }
        }

        let errors = self.calculate_errors(&perts).await?;
        let mut best_idx = 0;
        let mut min_error = self.stats.final_error;

        for (i, &err) in errors.iter().enumerate() {
            if err < min_error {
                min_error = err;
                best_idx = i;
            }
        }

        if best_idx > 0 {
            let pert = &perts[best_idx];
            if pert.prim_idx == 9999 {
                self.primitives.push(Primitive::new_sphere(
                    pert.pos_delta,
                    pert.size_scale,
                    if pert.op == 0 {
                        BooleanOp::Union
                    } else {
                        BooleanOp::Difference
                    },
                ));
            } else if pert.prim_idx < 8888 {
                let prim = &mut self.primitives[pert.prim_idx as usize];
                prim.pos = (prim.pos + pert.pos_delta).clamp(Vec3::ZERO, Vec3::ONE);
                prim.size =
                    (prim.size * pert.size_scale).clamp(Vec3::splat(0.001), Vec3::splat(0.5));
            }
        }

        self.stats.final_error = min_error;
        Ok(min_error)
    }

    async fn calculate_errors(&mut self, perts: &[Perturbation]) -> Result<Vec<f32>> {
        self.seed += 1;

        let config = Config {
            num_samples: self.samples_per_candidate,
            num_primitives: self.primitives.len() as u32,
            seed: self.seed,
            num_candidates: self.num_candidates,
        };

        self.queue
            .write_buffer(&self.config_buffer, 0, bytemuck::cast_slice(&[config]));
        if !self.primitives.is_empty() {
            self.queue.write_buffer(
                &self.primitive_buffer,
                0,
                bytemuck::cast_slice(&self.primitives),
            );
        }
        self.queue
            .write_buffer(&self.perturbation_buffer, 0, bytemuck::cast_slice(perts));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Optimizer Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.primitive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.target_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.results_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.config_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.perturbation_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.samples_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Error Calculation Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Error Calculation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                self.num_candidates,
                self.samples_per_candidate.div_ceil(256),
                1,
            );
        }

        let groups_per_cand = self.samples_per_candidate / 256;
        let total_results = self.num_candidates * groups_per_cand;
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.results_staging_buffer,
            0,
            (std::mem::size_of::<ErrorResult>() * total_results as usize) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.results_staging_buffer.slice(..);
        let (sender, receiver) =
            futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            let results: &[ErrorResult] = bytemuck::cast_slice(&data);

            self.last_results = results.to_vec();

            let mut cand_errors = Vec::with_capacity(self.num_candidates as usize);

            for c in 0..self.num_candidates as usize {
                let mut mse_total = 0.0;
                let mut iou_min_total = 0.0;
                let mut iou_max_total = 0.0;

                for g in 0..groups_per_cand as usize {
                    let res = &results[c * groups_per_cand as usize + g];
                    mse_total += res.mse_sum;
                    iou_min_total += res.iou_min_sum;
                    iou_max_total += res.iou_max_sum;
                }

                let mse = mse_total / self.samples_per_candidate as f32;
                let iou = if iou_max_total > 0.0 {
                    iou_min_total / iou_max_total
                } else {
                    1.0
                };

                let alpha = 0.5;
                let beta = 0.5;
                let error = alpha * mse + beta * (1.0 - iou);
                if c == 0 {
                    println!("DEBUG: Cand 0: mse={}, iou={}, error={}", mse, iou, error);
                }
                cand_errors.push(error);
            }

            drop(data);
            self.results_staging_buffer.unmap();

            Ok(cand_errors)
        } else {
            anyhow::bail!("Failed to read back results from GPU")
        }
    }

    pub fn generate_irmf(&self) -> String {
        let min = self.target_volume.min;
        let max = self.target_volume.max;
        let size = max - min;

        let notes = format!(
            "Generated by volume-to-irmf. Iterations: {}, Duration: {:?}, Final Error: {}. Suggestions: Increase iterations or primitive count for better detail.",
            self.stats.iterations, self.stats.duration, self.stats.final_error
        );

        let mut primitives_code = String::new();
        primitives_code.push_str("  var val = 0.0;\n");
        primitives_code.push_str("  let p_norm = (xyz - vec3f(");
        primitives_code.push_str(&format!("{}, {}, {}", min.x, min.y, min.z));
        primitives_code.push_str(")) / vec3f(");
        primitives_code.push_str(&format!("{}, {}, {}", size.x, size.y, size.z));
        primitives_code.push_str(");\n");

        for prim in &self.primitives {
            let prim_type = prim.prim_type;
            let op = prim.op;
            let p = prim.pos;
            let s = prim.size;

            let dist_func = if prim_type == 0 {
                // Sphere
                format!(
                    "length(p_norm - vec3f({}, {}, {})) - {}",
                    p.x, p.y, p.z, s.x
                )
            } else {
                // Cube
                format!(
                    "sd_box(p_norm - vec3f({}, {}, {}), vec3f({}, {}, {}))",
                    p.x, p.y, p.z, s.x, s.y, s.z
                )
            };

            let op_code = if op == 0 {
                // Union
                format!(
                    "  val = max(val, select(0.0, 1.0, {} <= 0.0));\n",
                    dist_func
                )
            } else {
                // Difference
                format!(
                    "  val = min(val, 1.0 - select(0.0, 1.0, {} <= 0.0));\n",
                    dist_func
                )
            };
            primitives_code.push_str(&op_code);
        }

        format!(
            r#"/*{{
  "irmf": "1.0",
  "language": "wgsl",
  "materials": ["Material"],
  "max": [{}, {}, {}],
  "min": [{}, {}, {}],
  "notes": {},
  "units": "mm"
}}*/

fn sd_box(p: vec3f, b: vec3f) -> f32 {{
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}}

fn mainModel4(xyz: vec3f) -> vec4f {{
{}
  return vec4f(val, 0.0, 0.0, 0.0);
}}
"#,
            max.x,
            max.y,
            max.z,
            min.x,
            min.y,
            min.z,
            serde_json::to_string(&notes).unwrap(),
            primitives_code
        )
    }
}
