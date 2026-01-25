use crate::primitives::{BooleanOp, Primitive};
use crate::volume::VoxelVolume;
use anyhow::Result;
use glam::Vec3;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ErrorResult {
    mse_sum: f32,
    iou_min_sum: f32,
    iou_max_sum: f32,
    sample_idx: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OctreeNode {
    pos: [f32; 4],
    size: [f32; 4],
    data: [f32; 4], // x: occupancy, y: level
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct OctreeConfig {
    dims: [u32; 4], // Explicit padding for WGSL vec3 alignment
    level: u32,
    threshold_low: f32,
    threshold_high: f32,
    max_nodes: u32,
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
    pad1: u32,
    pad2: u32,
    pad3: u32,
    pos_delta: [f32; 4],
    size_scale: [f32; 4],
    op: u32,
    pad4: u32,
    pad5: u32,
    pad6: u32,
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

    mipmap_pipeline: wgpu::ComputePipeline,
    octree_pipeline: wgpu::ComputePipeline,
    octree_bind_group_layout: wgpu::BindGroupLayout,

    num_candidates: u32,
    samples_per_candidate: u32,
    seed: u32,

    samples: Vec<[f32; 4]>,
    filled_voxels: Vec<Vec3>,
    last_results: Vec<ErrorResult>,
    last_best_error: f32,
    iterations_since_improvement: u32,

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

        let octree_shader_src = include_str!("octree.wgsl");
        let octree_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Octree Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(octree_shader_src)),
        });

        let octree_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Octree Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D3,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let octree_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Octree Pipeline Layout"),
                bind_group_layouts: &[&octree_bind_group_layout],
                push_constant_ranges: &[],
            });

        let mipmap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mipmap Pipeline"),
            layout: Some(&octree_pipeline_layout),
            module: &octree_shader,
            entry_point: Some("mipmap_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let octree_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Octree Extract Pipeline"),
            layout: Some(&octree_pipeline_layout),
            module: &octree_shader,
            entry_point: Some("extract_main"),
            compilation_options: Default::default(),
            cache: None,
        });

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
        let samples_per_candidate = 1024 * 128; // 128k
        let total_workgroups = num_candidates * (samples_per_candidate / 256);

        let samples_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Samples Buffer"),
            size: (std::mem::size_of::<[f32; 4]>() * samples_per_candidate as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut filled_voxels = Vec::new();
        for z in 0..target_volume.dims[2] {
            for y in 0..target_volume.dims[1] {
                for x in 0..target_volume.dims[0] {
                    if target_volume.get(x, y, z) > 0.5 {
                        filled_voxels.push(Vec3::new(
                            (x as f32 + 0.5) / target_volume.dims[0] as f32,
                            (y as f32 + 0.5) / target_volume.dims[1] as f32,
                            (z as f32 + 0.5) / target_volume.dims[2] as f32,
                        ));
                    }
                }
            }
        }

        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut samples = Vec::with_capacity(samples_per_candidate as usize);
        for i in 0..samples_per_candidate {
            let s = if !filled_voxels.is_empty() && (i % 2 == 0 || i < samples_per_candidate / 4) {
                filled_voxels[rng.gen_range(0..filled_voxels.len())]
            } else {
                Vec3::new(rng.r#gen(), rng.r#gen(), rng.r#gen())
            };
            samples.push([s.x, s.y, s.z, 0.0]);
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
            size: (std::mem::size_of::<Primitive>() * 2048) as u64,
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
            perturbation_buffer,
            results_buffer,
            results_staging_buffer,
            samples_buffer,
            mipmap_pipeline,
            octree_pipeline,
            octree_bind_group_layout,
            num_candidates,
            samples_per_candidate,
            seed: 0,
            samples,
            filled_voxels,
            last_results: Vec::new(),
            last_best_error: 1.0,
            iterations_since_improvement: 0,
            stats: Stats {
                iterations: 0,
                duration: std::time::Duration::from_secs(0),
                final_error: 1.0,
            },
            start_time: std::time::Instant::now(),
        })
    }

    pub fn add_primitive(&mut self, prim: Primitive) {
        self.primitives.push(prim);
    }

    /// Initializes primitives using a hierarchical octree-based algorithm.
    pub async fn octree_initialize(&mut self, target_count: usize) -> Result<()> {
        let [w, h, d] = self.target_volume.dims;
        let mut current_dims = [w, h, d];
        let mut pyramid = Vec::new();

        // Ensure we have dummy objects for bindings
        let dummy_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Pass 1: Generate Mipmap Pyramid
        let mut current_view = self.target_texture_view.clone();
        let mut level = 0;

        while current_dims[0] > 1 || current_dims[1] > 1 || current_dims[2] > 1 {
            let next_dims = [
                (current_dims[0] + 1) / 2,
                (current_dims[1] + 1) / 2,
                (current_dims[2] + 1) / 2,
            ];

            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Mipmap Level {}", level + 1)),
                size: wgpu::Extent3d {
                    width: next_dims[0],
                    height: next_dims[1],
                    depth_or_array_layers: next_dims[2],
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            let config = OctreeConfig {
                dims: [current_dims[0], current_dims[1], current_dims[2], 0],
                level,
                threshold_low: 0.05,
                threshold_high: 0.95,
                max_nodes: 10000,
            };
            let config_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Octree Config"),
                    contents: bytemuck::cast_slice(&[config]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let dummy_nodes = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Nodes"),
                size: 1024,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            let dummy_count = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Dummy Count"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mipmap Bind Group"),
                layout: &self.octree_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&current_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dummy_nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: dummy_count.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut compute_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                compute_pass.set_pipeline(&self.mipmap_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(
                    next_dims[0].div_ceil(8),
                    next_dims[1].div_ceil(8),
                    next_dims[2],
                );
            }
            self.queue.submit(Some(encoder.finish()));

            pyramid.push((view.clone(), next_dims, level + 1));
            current_view = view;
            current_dims = next_dims;
            level += 1;
        }

        // Pass 2: Extract Primitives (Multi-level)
        let mut all_potential_nodes = Vec::new();
        let max_extracted = 1000000; // Increased to 1M to avoid bottom-slice bias
        let node_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Octree Node Output"),
            size: (std::mem::size_of::<OctreeNode>() * max_extracted) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Octree Count Output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_node_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Node Buffer"),
            size: (std::mem::size_of::<OctreeNode>() * max_extracted) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_count_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut level_textures = pyramid;
        level_textures.insert(
            0,
            (self.target_texture_view.clone(), self.target_volume.dims, 0),
        );

        for (view, dims, lvl) in level_textures {
            self.queue
                .write_buffer(&count_buffer, 0, bytemuck::cast_slice(&[0u32]));
            // Lower thresholds ensure sparse wires are captured at coarse levels.
            let config = OctreeConfig {
                dims: [w, h, d, 0],
                level: lvl,
                threshold_low: 0.01,
                threshold_high: 0.02, // Lowered from 0.1 to capture thin features like Rodin coil wires
                max_nodes: max_extracted as u32,
            };
            let config_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Octree Extract Config"),
                    contents: bytemuck::cast_slice(&[config]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Extract Bind Group"),
                layout: &self.octree_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&dummy_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: node_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: count_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut compute_pass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                compute_pass.set_pipeline(&self.octree_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(dims[0].div_ceil(8), dims[1].div_ceil(8), dims[2]);
            }
            encoder.copy_buffer_to_buffer(
                &node_buffer,
                0,
                &staging_node_buffer,
                0,
                (std::mem::size_of::<OctreeNode>() * max_extracted) as u64,
            );
            encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_count_buffer, 0, 4);
            self.queue.submit(Some(encoder.finish()));

            let (tx, rx) = futures::channel::oneshot::channel();
            staging_count_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |v| {
                    let _ = tx.send(v);
                });
            self.device.poll(wgpu::Maintain::Wait);
            rx.await??;
            let count =
                bytemuck::cast_slice::<u8, u32>(&staging_count_buffer.slice(..).get_mapped_range())
                    [0] as usize;
            staging_count_buffer.unmap();

            if count > 0 {
                let (tx, rx) = futures::channel::oneshot::channel();
                staging_node_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |v| {
                        let _ = tx.send(v);
                    });
                self.device.poll(wgpu::Maintain::Wait);
                rx.await??;
                let mapped = staging_node_buffer.slice(..).get_mapped_range();
                let nodes: &[OctreeNode] = bytemuck::cast_slice(&mapped);
                for n in &nodes[..count.min(max_extracted)] {
                    all_potential_nodes.push(*n);
                }
                drop(mapped);
                staging_node_buffer.unmap();
            }
        }

        // Randomize then stable sort to randomize ties in importance.
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        all_potential_nodes.shuffle(&mut rng);

        all_potential_nodes.sort_by(|a, b| {
            let imp_a = a.data[0] * (1u32 << (3 * a.data[1] as u32)) as f32;
            let imp_b = b.data[0] * (1u32 << (3 * b.data[1] as u32)) as f32;
            imp_b.partial_cmp(&imp_a).unwrap()
        });

        self.primitives.clear();
        for n in all_potential_nodes.iter().take(target_count) {
            self.primitives.push(Primitive::new_cube(
                Vec3::new(n.pos[0], n.pos[1], n.pos[2]),
                Vec3::new(n.size[0], n.size[1], n.size[2]),
                BooleanOp::Union,
            ));
        }

        if !self.primitives.is_empty() {
            println!("First 5 extracted primitives:");
            for i in 0..5.min(self.primitives.len()) {
                let n = &all_potential_nodes[i];
                println!(
                    "  Level {}: pos={:?}, size={:?}, occupancy={}",
                    n.data[1],
                    Vec3::new(n.pos[0], n.pos[1], n.pos[2]),
                    Vec3::new(n.size[0], n.size[1], n.size[2]),
                    n.data[0]
                );
            }
        }

        println!(
            "Octree initialization produced {} primitives from {} candidates.",
            self.primitives.len(),
            all_potential_nodes.len()
        );
        Ok(())
    }

    /// Initializes primitives using a greedy box-growing algorithm to cover all filled voxels.
    pub fn greedy_box_initialize(&mut self) {
        let [w, h, d] = self.target_volume.dims;
        let mut covered = vec![false; (w * h * d) as usize];
        let mut primitives = Vec::new();

        let mut v_min = Vec3::splat(f32::MAX);
        let mut v_max = Vec3::splat(f32::MIN);
        let mut filled_count = 0;

        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let idx = ((z * h + y) * w + x) as usize;
                    if self.target_volume.data[idx] > 0.5 {
                        filled_count += 1;
                        let p = Vec3::new(x as f32, y as f32, z as f32);
                        v_min = v_min.min(p);
                        v_max = v_max.max(p);

                        if !covered[idx] {
                            let mut dx = 0;
                            while x + dx + 1 < w {
                                let next_idx = idx + (dx + 1) as usize;
                                if self.target_volume.data[next_idx] > 0.5 && !covered[next_idx] {
                                    dx += 1;
                                } else {
                                    break;
                                }
                            }

                            let mut dy = 0;
                            'y_loop: while y + dy + 1 < h {
                                for i in 0..=dx {
                                    let next_idx =
                                        (((z * h + (y + dy + 1)) * w) + (x + i)) as usize;
                                    if !(self.target_volume.data[next_idx] > 0.5
                                        && !covered[next_idx])
                                    {
                                        break 'y_loop;
                                    }
                                }
                                dy += 1;
                            }

                            let mut dz = 0;
                            'z_loop: while z + dz + 1 < d {
                                for j in 0..=dy {
                                    for i in 0..=dx {
                                        let next_idx =
                                            ((((z + dz + 1) * h + (y + j)) * w) + (x + i)) as usize;
                                        if !(self.target_volume.data[next_idx] > 0.5
                                            && !covered[next_idx])
                                        {
                                            break 'z_loop;
                                        }
                                    }
                                }
                                dz += 1;
                            }

                            let pos = Vec3::new(
                                (x as f32 + (dx as f32 + 1.0) / 2.0) / w as f32,
                                (y as f32 + (dy as f32 + 1.0) / 2.0) / h as f32,
                                (z as f32 + (dz as f32 + 1.0) / 2.0) / d as f32,
                            );
                            let size = Vec3::new(
                                (dx as f32 + 1.0) / 2.0 / w as f32,
                                (dy as f32 + 1.0) / 2.0 / h as f32,
                                (dz as f32 + 1.0) / 2.0 / d as f32,
                            );

                            primitives.push(Primitive::new_cube(pos, size, BooleanOp::Union));

                            for k in 0..=dz {
                                for j in 0..=dy {
                                    for i in 0..=dx {
                                        let c_idx =
                                            ((((z + k) * h + (y + j)) * w) + (x + i)) as usize;
                                        covered[c_idx] = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!(
            "Greedy complete: Filled={}, VoxelBounds={:?} to {:?}, Prims={}",
            filled_count,
            v_min,
            v_max,
            primitives.len()
        );

        let mut p_min = Vec3::splat(f32::MAX);
        let mut p_max = Vec3::splat(f32::MIN);
        for p in &primitives {
            let pos = Vec3::from_slice(&p.pos[..3]);
            let size = Vec3::from_slice(&p.size[..3]);
            p_min = p_min.min(pos - size);
            p_max = p_max.max(pos + size);
        }
        println!("Primitive normalized bounds: {:?} to {:?}", p_min, p_max);

        self.primitives = primitives;
    }

    /// Reduces the number of primitives by merging those that minimize introduced error.
    pub fn decimate(&mut self, target_count: usize) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        println!(
            "Decimating {} to {}...",
            self.primitives.len(),
            target_count
        );

        let mut dist_threshold_sq = 0.0001;
        let mut min_efficiency = 0.5;

        while self.primitives.len() > target_count {
            // Sort by X position occasionally to keep neighbors close in the list
            if self.primitives.len() % 1000 == 0 {
                self.primitives.sort_by(|a, b| a.pos[0].partial_cmp(&b.pos[0]).unwrap());
            }

            let mut best_pair = (0, 0);
            let mut min_cost = f32::MAX;

            let samples = if self.primitives.len() > 10000 {
                2000
            } else {
                5000
            };

            for _ in 0..samples {
                let i = rng.gen_range(0..self.primitives.len());
                // Pick a neighbor in the sorted list with high probability
                let j = if rng.gen_bool(0.9) {
                    let offset = rng.gen_range(1..20);
                    if rng.gen_bool(0.5) {
                        if i >= offset { i - offset } else { (i + offset).min(self.primitives.len() - 1) }
                    } else {
                        (i + offset).min(self.primitives.len() - 1)
                    }
                } else {
                    rng.gen_range(0..self.primitives.len())
                };

                if i == j {
                    continue;
                }

                let p1 = &self.primitives[i];
                let p2 = &self.primitives[j];

                let p1_pos = Vec3::from_slice(&p1.pos[..3]);
                let p2_pos = Vec3::from_slice(&p2.pos[..3]);
                let p1_size = Vec3::from_slice(&p1.size[..3]);
                let p2_size = Vec3::from_slice(&p2.size[..3]);

                let dist_sq = (p1_pos - p2_pos).length_squared();
                if dist_sq > dist_threshold_sq {
                    continue;
                }

                let combined_min = (p1_pos - p1_size).min(p2_pos - p2_size);
                let combined_max = (p1_pos + p1_size).max(p2_pos + p2_size);
                let combined_size = (combined_max - combined_min) / 2.0;
                let combined_pos = (combined_min + combined_max) / 2.0;

                let mut filled_count = 0;
                let test_samples = 32; // Increased for better accuracy
                for _ in 0..test_samples {
                    let rp = combined_pos
                        + combined_size
                            * Vec3::new(
                                rng.gen_range(-1.0..1.0),
                                rng.gen_range(-1.0..1.0),
                                rng.gen_range(-1.0..1.0),
                            );
                    let vx = (rp.x * self.target_volume.dims[0] as f32) as u32;
                    let vy = (rp.y * self.target_volume.dims[1] as f32) as u32;
                    let vz = (rp.z * self.target_volume.dims[2] as f32) as u32;
                    if self.target_volume.get(vx, vy, vz) > 0.5 {
                        filled_count += 1;
                    }
                }

                let efficiency = filled_count as f32 / test_samples as f32;
                let combined_vol = combined_size.x * combined_size.y * combined_size.z;
                let cost = (1.0 - efficiency) * combined_vol;

                if efficiency >= min_efficiency && cost < min_cost {
                    min_cost = cost;
                    best_pair = (i, j);
                }
            }

            if min_cost == f32::MAX {
                dist_threshold_sq *= 1.2;
                if dist_threshold_sq > 0.1 { // Cap distance threshold
                    dist_threshold_sq = 0.1;
                    min_efficiency *= 0.9; // Lower efficiency requirement if stuck
                    if min_efficiency < 0.01 {
                        println!(
                            "Gave up on efficiency. Stopping at {}.",
                            self.primitives.len()
                        );
                        break;
                    }
                }
                continue;
            }

            let (idx1, idx2) = if best_pair.0 > best_pair.1 {
                (best_pair.0, best_pair.1)
            } else {
                (best_pair.1, best_pair.0)
            };
            let p1 = self.primitives.remove(idx1);
            let p2 = self.primitives.remove(idx2);

            let p1_pos = Vec3::from_slice(&p1.pos[..3]);
            let p2_pos = Vec3::from_slice(&p2.pos[..3]);
            let p1_size = Vec3::from_slice(&p1.size[..3]);
            let p2_size = Vec3::from_slice(&p2.size[..3]);

            let combined_min = (p1_pos - p1_size).min(p2_pos - p2_size);
            let combined_max = (p1_pos + p1_size).max(p2_pos + p2_size);

            self.primitives.push(Primitive::new_cube(
                (combined_min + combined_max) / 2.0,
                (combined_max - combined_min) / 2.0,
                BooleanOp::Union,
            ));

            if self.primitives.len() % 1000 == 0 {
                println!(
                    "... remaining: {}, dist_thresh_sq: {:.6}",
                    self.primitives.len(),
                    dist_threshold_sq
                );
            }
        }
    }

    pub async fn run_iteration(&mut self) -> Result<f32> {
        self.stats.iterations += 1;
        self.stats.duration = self.start_time.elapsed();

        use rand::Rng;
        let mut rng = rand::thread_rng();

        self.samples.clear();
        for i in 0..self.samples_per_candidate {
            let s = if !self.filled_voxels.is_empty()
                && (i % 2 == 0 || i < self.samples_per_candidate / 4)
            {
                self.filled_voxels[rng.gen_range(0..self.filled_voxels.len())]
            } else {
                Vec3::new(rng.r#gen(), rng.r#gen(), rng.r#gen())
            };
            self.samples.push([s.x, s.y, s.z, 0.0]);
        }
        self.queue
            .write_buffer(&self.samples_buffer, 0, bytemuck::cast_slice(&self.samples));

        let mut seed_positions = Vec::new();
        if !self.last_results.is_empty() {
            let groups_per_cand = self.samples_per_candidate / 256;
            let mut errors_with_idx: Vec<(f32, usize)> = self.last_results
                [0..groups_per_cand as usize]
                .iter()
                .enumerate()
                .map(|(i, r)| (r.mse_sum, i))
                .collect();
            errors_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

            for i in 0..10.min(errors_with_idx.len()) {
                let group_idx = errors_with_idx[i].1;
                let sample_idx = group_idx as u32 * 256 + rng.gen_range(0..256);
                let s = self.samples[sample_idx as usize];
                seed_positions.push(Vec3::new(s[0], s[1], s[2]));
            }
        }
        if seed_positions.is_empty() {
            seed_positions.push(Vec3::new(rng.r#gen(), rng.r#gen(), rng.r#gen()));
        }

        let mut perts = Vec::with_capacity(self.num_candidates as usize);
        perts.push(Perturbation {
            prim_idx: 8888,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            pos_delta: [0.0, 0.0, 0.0, 0.0],
            size_scale: [1.0, 1.0, 1.0, 0.0],
            op: 0,
            pad4: 0,
            pad5: 0,
            pad6: 0,
        });

        let seed_pos = seed_positions[rng.gen_range(0..seed_positions.len())];
        perts.push(Perturbation {
            prim_idx: 9999,
            pad1: 0,
            pad2: 0,
            pad3: 0,
            pos_delta: [seed_pos.x, seed_pos.y, seed_pos.z, 0.0],
            size_scale: [0.01, 0.01, 0.01, 0.0],
            op: 0,
            pad4: 0,
            pad5: 0,
            pad6: 0,
        });

        for _i in 2..self.num_candidates as usize {
            let seed_pos = seed_positions[rng.gen_range(0..seed_positions.len())];
            if self.primitives.len() < 2048 && (self.primitives.is_empty() || rng.gen_bool(0.3)) {
                let size = rng.gen_range(0.005..0.05);
                let aspect = Vec3::new(
                    rng.gen_range(0.5..2.0),
                    rng.gen_range(0.5..2.0),
                    rng.gen_range(0.5..2.0),
                );
                let d = seed_pos
                    + Vec3::new(
                        rng.gen_range(-0.01..0.01),
                        rng.gen_range(-0.01..0.01),
                        rng.gen_range(-0.01..0.01),
                    );
                let s = Vec3::splat(size) * aspect;
                perts.push(Perturbation {
                    prim_idx: 9999,
                    pad1: 0,
                    pad2: 0,
                    pad3: 0,
                    pos_delta: [d.x, d.y, d.z, 0.0],
                    size_scale: [s.x, s.y, s.z, 0.0],
                    op: rng.gen_range(0..4),
                    pad4: 0,
                    pad5: 0,
                    pad6: 0,
                });
            } else if !self.primitives.is_empty() {
                let prim_idx = rng.gen_range(0..self.primitives.len()) as u32;
                if rng.gen_bool(0.1) {
                    let prim_pos = Vec3::from_slice(&self.primitives[prim_idx as usize].pos[..3]);
                    let prim_size = Vec3::from_slice(&self.primitives[prim_idx as usize].size[..3]);
                    let d = seed_pos - prim_pos;
                    let s = Vec3::splat(rng.gen_range(0.005..0.05)) / prim_size;
                    perts.push(Perturbation {
                        prim_idx,
                        pad1: 0,
                        pad2: 0,
                        pad3: 0,
                        pos_delta: [d.x, d.y, d.z, 0.0],
                        size_scale: [s.x, s.y, s.z, 0.0],
                        op: rng.gen_range(0..4),
                        pad4: 0,
                        pad5: 0,
                        pad6: 0,
                    });
                } else {
                    let d = Vec3::new(
                        rng.gen_range(-0.02..0.02),
                        rng.gen_range(-0.02..0.02),
                        rng.gen_range(-0.02..0.02),
                    );
                    let s = Vec3::new(
                        rng.gen_range(0.9..1.1),
                        rng.gen_range(0.9..1.1),
                        rng.gen_range(0.9..1.1),
                    );
                    perts.push(Perturbation {
                        prim_idx,
                        pad1: 0,
                        pad2: 0,
                        pad3: 0,
                        pos_delta: [d.x, d.y, d.z, 0.0],
                        size_scale: [s.x, s.y, s.z, 0.0],
                        op: self.primitives[prim_idx as usize].op,
                        pad4: 0,
                        pad5: 0,
                        pad6: 0,
                    });
                }
            } else {
                perts.push(perts[0]);
            }
        }

        let errors = self.calculate_errors(&perts).await?;
        let identity_error = errors[0];
        let mut best_idx = 0;
        let mut min_error = identity_error;
        let mut improved_count = 0;

        for (i, &err) in errors.iter().enumerate() {
            if err < identity_error {
                improved_count += 1;
            }
            if err < min_error {
                min_error = err;
                best_idx = i;
            }
        }

        if self.stats.iterations % 10 == 0 {
            println!(
                "Iteration {}: best_error = {}, improved = {}/{}",
                self.stats.iterations, min_error, improved_count, self.num_candidates
            );
        }

        if best_idx > 0 {
            let pert = &perts[best_idx];
            if pert.prim_idx == 9999 {
                let op = match pert.op {
                    0 | 2 => BooleanOp::Union,
                    _ => BooleanOp::Difference,
                };
                let d = Vec3::from_slice(&pert.pos_delta[..3]);
                let s = Vec3::from_slice(&pert.size_scale[..3]);
                if pert.op >= 2 {
                    self.primitives.push(Primitive::new_cube(d, s, op));
                } else {
                    self.primitives.push(Primitive::new_sphere(d, s.x, op));
                }
            } else if pert.prim_idx < 8888 {
                let prim = &mut self.primitives[pert.prim_idx as usize];
                let prim_pos = Vec3::from_slice(&prim.pos[..3]);
                let prim_size = Vec3::from_slice(&prim.size[..3]);
                let pos_delta = Vec3::from_slice(&pert.pos_delta[..3]);
                let size_scale = Vec3::from_slice(&pert.size_scale[..3]);

                let new_pos = (prim_pos + pos_delta).clamp(Vec3::ZERO, Vec3::ONE);
                let new_size =
                    (prim_size * size_scale).clamp(Vec3::splat(0.0005), Vec3::splat(0.5));

                prim.pos = [new_pos.x, new_pos.y, new_pos.z, 0.0];
                prim.size = [new_size.x, new_size.y, new_size.z, 0.0];

                if pert.op < 4 && (pos_delta.length() > 0.5 || size_scale.min_element() < 0.1) {
                    prim.prim_type = if pert.op >= 2 { 1 } else { 0 };
                    prim.op = pert.op % 2;
                }
            }
        }

        if min_error < self.last_best_error * 0.999 {
            self.last_best_error = min_error;
            self.iterations_since_improvement = 0;
        } else {
            self.iterations_since_improvement += 1;
        }

        if self.iterations_since_improvement > 50 && self.primitives.len() < 2048 {
            let seed_pos = seed_positions[rng.gen_range(0..seed_positions.len())];
            self.primitives.push(Primitive::new_sphere(
                seed_pos,
                rng.gen_range(0.01..0.03),
                BooleanOp::Union,
            ));
            self.iterations_since_improvement = 0;
            println!("Forced adding primitive due to no improvement.");
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
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut compute_pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(
                self.num_candidates,
                self.samples_per_candidate / 256,
                1,
            );
        }
        let total_results = self.num_candidates * (self.samples_per_candidate / 256);
        encoder.copy_buffer_to_buffer(
            &self.results_buffer,
            0,
            &self.results_staging_buffer,
            0,
            (std::mem::size_of::<ErrorResult>() * total_results as usize) as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = futures::channel::oneshot::channel();
        self.results_staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |v| {
                let _ = tx.send(v);
            });
        self.device.poll(wgpu::Maintain::Wait);
        rx.await??;

        let mapped = self.results_staging_buffer.slice(..).get_mapped_range();
        let results: &[ErrorResult] = bytemuck::cast_slice(&mapped);
        self.last_results = results.to_vec();
        let mut cand_errors = Vec::with_capacity(self.num_candidates as usize);
        let groups_per_cand = self.samples_per_candidate / 256;
        for c in 0..self.num_candidates as usize {
            let mut mse_total = 0.0;
            for g in 0..groups_per_cand as usize {
                mse_total += results[c * groups_per_cand as usize + g].mse_sum;
            }
            cand_errors.push(mse_total / self.samples_per_candidate as f32);
        }
        drop(mapped);
        self.results_staging_buffer.unmap();
        Ok(cand_errors)
    }

    pub fn generate_irmf(&self) -> String {
        let min = self.target_volume.min;
        let max = self.target_volume.max;
        let size = max - min;
        let notes = format!(
            "Generated by volume-to-irmf. Iterations: {}, Primitives: {}, Error: {}",
            self.stats.iterations,
            self.primitives.len(),
            self.stats.final_error
        );
        let mut primitives_code = String::new();
        primitives_code.push_str("  var val = 0.0;\n");
        primitives_code.push_str(&format!(
            "  let p_norm = (xyz - vec3f({:.4}, {:.4}, {:.4})) / vec3f({:.4}, {:.4}, {:.4});\n",
            min.x, min.y, min.z, size.x, size.y, size.z
        ));
        for prim in &self.primitives {
            let df = if prim.prim_type == 0 {
                format!(
                    "length(p_norm - vec3f({:.4}, {:.4}, {:.4})) - {:.4}",
                    prim.pos[0], prim.pos[1], prim.pos[2], prim.size[0]
                )
            } else {
                format!(
                    "sd_box(p_norm - vec3f({:.4}, {:.4}, {:.4}), vec3f({:.4}, {:.4}, {:.4}))",
                    prim.pos[0], prim.pos[1], prim.pos[2], prim.size[0], prim.size[1], prim.size[2]
                )
            };
            if prim.op == 0 {
                primitives_code.push_str(&format!(
                    "  val = max(val, clamp(0.5 - ({}) * 100.0, 0.0, 1.0));\n",
                    df
                ));
            } else {
                primitives_code.push_str(&format!(
                    "  val = min(val, 1.0 - clamp(0.5 - ({}) * 100.0, 0.0, 1.0));\n",
                    df
                ));
            }
        }
        format!(
            r#"/*{{
  "irmf": "1.0",
  "language": "wgsl",
  "materials": ["Material"],
  "max": [{:.4}, {:.4}, {:.4}],
  "min": [{:.4}, {:.4}, {:.4}],
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
