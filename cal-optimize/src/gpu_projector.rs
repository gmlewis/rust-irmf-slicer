use crate::Projector;
use ndarray::Array3;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub struct GpuProjector {
    device: wgpu::Device,
    queue: wgpu::Queue,
    forward_pipeline: wgpu::ComputePipeline,
    backward_pipeline: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    nr: u32,
    n_angles: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    angles: [[f32; 4]; 128], // Packed angles for WGSL
}

impl GpuProjector {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create WGPU device");

        let forward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Forward Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("forward.wgsl"))),
        });

        let backward_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Backward Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("backward.wgsl"))),
        });

        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Forward Pipeline"),
            layout: None,
            module: &forward_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let backward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Backward Pipeline"),
            layout: None,
            module: &backward_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            forward_pipeline,
            backward_pipeline,
        }
    }
}

impl Projector for GpuProjector {
    fn forward(&self, volume: &Array3<f32>, angles: &[f32]) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let n_angles = angles.len();
        let nr = ((nx as f32).powi(2) + (ny as f32).powi(2)).sqrt().ceil() as usize;

        let mut packed_angles = [[0.0f32; 4]; 128];
        for (i, &a) in angles.iter().enumerate().take(128) {
            packed_angles[i][0] = a.to_radians();
        }

        let params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            nr: nr as u32,
            n_angles: n_angles as u32,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            angles: packed_angles,
        };

        let volume_data: Vec<f32> = volume.as_standard_layout().iter().cloned().collect();
        let vol_max = volume_data.iter().fold(0.0f32, |m, &v| m.max(v));
        println!(
            "GPU Forward: volume max: {:.4}, nr: {}, n_angles: {}, nz: {}",
            vol_max, nr, n_angles, nz
        );

        let volume_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Volume Buffer"),
                contents: bytemuck::cast_slice(&volume_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let projections_size = (nr * n_angles * nz * 4) as u64;
        let projections_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Projections Buffer"),
            size: projections_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout = self.forward_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: volume_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: projections_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.forward_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(
                (nr as u32).div_ceil(8),
                (n_angles as u32).div_ceil(8),
                nz as u32,
            );
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: projections_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&projections_buffer, 0, &staging_buffer, 0, projections_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Buffer mapping failed");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Array3::from_shape_vec((nr, n_angles, nz), result).unwrap()
    }

    fn backward(
        &self,
        projections: &Array3<f32>,
        angles: &[f32],
        target_dim: (usize, usize, usize),
    ) -> Array3<f32> {
        let (nr, n_angles, nz) = projections.dim();
        let (nx, ny, _nz_target) = target_dim;

        let mut packed_angles = [[0.0f32; 4]; 128];
        for (i, &a) in angles.iter().enumerate().take(128) {
            packed_angles[i][0] = a.to_radians();
        }

        let params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            nr: nr as u32,
            n_angles: n_angles as u32,
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            angles: packed_angles,
        };

        let projections_data: Vec<f32> = projections.as_standard_layout().iter().cloned().collect();
        let proj_max = projections_data.iter().fold(0.0f32, |m, &v| m.max(v));
        println!("GPU Backward: projections max: {:.4}", proj_max);

        let projections_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Projections Buffer"),
                    contents: bytemuck::cast_slice(&projections_data),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let volume_size = (nx * ny * nz * 4) as u64;
        let volume_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volume Buffer"),
            size: volume_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout = self.backward_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: projections_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: volume_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: None,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.backward_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((nx as u32).div_ceil(8), (ny as u32).div_ceil(8), nz as u32);
        }

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: volume_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&volume_buffer, 0, &staging_buffer, 0, volume_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().expect("Buffer mapping failed");

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Array3::from_shape_vec((nx, ny, nz), result).unwrap()
    }
}
