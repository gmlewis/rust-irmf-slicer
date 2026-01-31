use anyhow::Result;
use std::sync::Arc;
use winit::window::Window;

pub struct ProjectionWindow {
    _window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    _config: wgpu::SurfaceConfiguration,
    _size: winit::dpi::PhysicalSize<u32>,

    // Textures for each projection frame
    _textures: Vec<wgpu::Texture>,
    bind_groups: Vec<wgpu::BindGroup>,
    render_pipeline: wgpu::RenderPipeline,
}

impl ProjectionWindow {
    pub async fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        width: u32,
        height: u32,
        monitor_index: Option<usize>,
    ) -> Result<Self> {
        let mut window_attributes = Window::default_attributes()
            .with_title("CAL Projection Window")
            .with_inner_size(winit::dpi::PhysicalSize::new(width, height));

        if let Some(idx) = monitor_index {
            let monitor = event_loop
                .available_monitors()
                .nth(idx)
                .ok_or_else(|| anyhow::anyhow!("Monitor index {} not found", idx))?;
            println!("Selecting monitor: {:?}", monitor.name());
            window_attributes = window_attributes
                .with_fullscreen(Some(winit::window::Fullscreen::Borderless(Some(monitor))));
        }

        let window = Arc::new(event_loop.create_window(window_attributes)?);

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to find a suitable GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width, size.height)
            .unwrap();
        surface.configure(&device, &config);

        // Simple shader for displaying a texture
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Projection Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            _window: window,
            surface,
            device,
            queue,
            _config: config,
            _size: size,
            _textures: Vec::new(),
            bind_groups: Vec::new(),
            render_pipeline,
        })
    }

    pub fn prepare_projections(&mut self, projections: &ndarray::Array3<f32>) {
        let (nr, n_angles, nz) = projections.dim();

        // Find global max value for normalization across all angles/frames
        let mut global_max = 0.0f32;
        for val in projections.iter() {
            if *val > global_max {
                global_max = *val;
            }
        }
        println!("Global max projection intensity: {:.4}", global_max);

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        for a_idx in 0..n_angles {
            let texture_size = wgpu::Extent3d {
                width: nr as u32,
                height: nz as u32,
                depth_or_array_layers: 1,
            };

            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                size: texture_size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                label: Some(&format!("Projection Texture {}", a_idx)),
                view_formats: &[],
            });

            let mut data = vec![0.0f32; nr * nz];
            for z in 0..nz {
                for r in 0..nr {
                    let val = projections[[r, a_idx, z]];
                    data[z * nr + r] = if global_max > 0.0 {
                        val / global_max
                    } else {
                        0.0
                    };
                }
            }

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * nr as u32),
                    rows_per_image: Some(nz as u32),
                },
                texture_size,
            );

            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &self.render_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: None,
            });

            self._textures.push(texture);
            self.bind_groups.push(bind_group);
        }
    }

    pub fn render(&self, frame_idx: usize) -> Result<()> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_groups[frame_idx], &[]);
            render_pass.draw(0..3, 0..1); // Draw a full-screen triangle
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    pub fn request_redraw(&self) {
        self._window.request_redraw();
    }
}
