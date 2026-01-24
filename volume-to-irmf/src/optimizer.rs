use crate::primitives::{BooleanOp, Primitive};
use crate::volume::VoxelVolume;
use anyhow::Result;
use glam::Vec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

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
    results_buffer: wgpu::Buffer,
    results_staging_buffer: wgpu::Buffer,
    
    num_samples: u32,
    seed: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Config {
    num_samples: u32,
    num_primitives: u32,
    seed: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ErrorResult {
    mse_sum: f32,
    iou_min_sum: f32,
    iou_max_sum: f32,
    _padding: f32,
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
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Create target volume texture
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

        let target_texture_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Load shader
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

        let num_samples = 1024 * 64; // Example sample count
        let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Config Buffer"),
            size: std::mem::size_of::<Config>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let primitive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Primitive Buffer"),
            size: (std::mem::size_of::<Primitive>() * 1024) as u64, // Max 1024 primitives for now
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let results_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Buffer"),
            size: (std::mem::size_of::<ErrorResult>() * num_samples as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let results_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Results Staging Buffer"),
            size: (std::mem::size_of::<ErrorResult>() * num_samples as usize) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            target_volume: Arc::new(target_volume),
            primitives: Vec::new(),
            pipeline,
            bind_group_layout,
            target_texture_view,
            config_buffer,
            primitive_buffer,
            results_buffer,
            results_staging_buffer,
            num_samples,
            seed: 0,
        })
    }

    pub fn add_primitive(&mut self, prim: Primitive) {
        self.primitives.push(prim);
    }

    pub async fn run_iteration(&mut self) -> Result<f32> {
        let current_error = self.calculate_error().await?;
        
        // Simple stochastic refinement
        // 1. Try to add a new primitive if we have room
        if self.primitives.len() < 1024 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let new_prim = if rng.gen_bool(0.5) {
                Primitive::new_sphere(
                    Vec3::new(rng.r#gen::<f32>(), rng.r#gen::<f32>(), rng.r#gen::<f32>()),
                    rng.gen_range(0.01..0.2),
                    if rng.gen_bool(0.8) { BooleanOp::Union } else { BooleanOp::Difference },
                )
            } else {
                Primitive::new_cube(
                    Vec3::new(rng.r#gen::<f32>(), rng.r#gen::<f32>(), rng.r#gen::<f32>()),
                    Vec3::new(rng.gen_range(0.01..0.2), rng.gen_range(0.01..0.2), rng.gen_range(0.01..0.2)),
                    if rng.gen_bool(0.8) { BooleanOp::Union } else { BooleanOp::Difference },
                )
            };

            self.primitives.push(new_prim);
            let new_error = self.calculate_error().await?;
            if new_error < current_error {
                return Ok(new_error);
            } else {
                self.primitives.pop();
            }
        }

        // 2. Try to refine existing primitives
        if !self.primitives.is_empty() {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let idx = rng.gen_range(0..self.primitives.len());
            let old_prim = self.primitives[idx];
            
            // Perturb
            self.primitives[idx].pos += Vec3::new(rng.gen_range(-0.05..0.05), rng.gen_range(-0.05..0.05), rng.gen_range(-0.05..0.05));
            self.primitives[idx].size *= rng.gen_range(0.9..1.1);
            
            let new_error = self.calculate_error().await?;
            if new_error < current_error {
                return Ok(new_error);
            } else {
                self.primitives[idx] = old_prim;
            }
        }

        Ok(current_error)
    }

    async fn calculate_error(&mut self) -> Result<f32> {
        self.seed += 1;
        
        let config = Config {
            num_samples: self.num_samples,
            num_primitives: self.primitives.len() as u32,
            seed: self.seed,
            _padding: 0,
        };
        
        self.queue.write_buffer(&self.config_buffer, 0, bytemuck::cast_slice(&[config]));
        if !self.primitives.is_empty() {
            self.queue.write_buffer(&self.primitive_buffer, 0, bytemuck::cast_slice(&self.primitives));
        }

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
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Error Calculation Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Error Calculation Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((self.num_samples + 63) / 64, 1, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.results_buffer, 0,
            &self.results_staging_buffer, 0,
            (std::mem::size_of::<ErrorResult>() * self.num_samples as usize) as u64,
        );

        self.queue.submit(Some(encoder.finish()));

        // Read back results
        let buffer_slice = self.results_staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            let results: &[ErrorResult] = bytemuck::cast_slice(&data);
            
            let mut mse_total = 0.0;
            let mut iou_min_total = 0.0;
            let mut iou_max_total = 0.0;
            
            for res in results {
                mse_total += res.mse_sum;
                iou_min_total += res.iou_min_sum;
                iou_max_total += res.iou_max_sum;
            }
            
            drop(data);
            self.results_staging_buffer.unmap();
            
            let mse = mse_total / self.num_samples as f32;
            let iou = if iou_max_total > 0.0 { iou_min_total / iou_max_total } else { 1.0 };
            
            // Reward function MSE + IoU Hybrid
            let alpha = 0.5;
            let beta = 0.5;
            let error = alpha * mse + beta * (1.0 - iou);
            
            Ok(error)
        } else {
            anyhow::bail!("Failed to read back results from GPU")
        }
    }

    pub fn generate_irmf(&self) -> String {
        let min = self.target_volume.min;
        let max = self.target_volume.max;
        let size = max - min;

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

            let dist_func = if prim_type == 0 { // Sphere
                format!("length(p_norm - vec3f({}, {}, {})) - {}", p.x, p.y, p.z, s.x)
            } else { // Cube
                format!("sd_box(p_norm - vec3f({}, {}, {}), vec3f({}, {}, {}))", p.x, p.y, p.z, s.x, s.y, s.z)
            };

            let op_code = if op == 0 { // Union
                format!("  val = max(val, select(0.0, 1.0, {} <= 0.0));\n", dist_func)
            } else { // Difference
                format!("  val = min(val, 1.0 - select(0.0, 1.0, {} <= 0.0));\n", dist_func)
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
            max.x, max.y, max.z, min.x, min.y, min.z, primitives_code
        )
    }
}
