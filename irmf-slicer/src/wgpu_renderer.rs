use crate::irmf::IrmfModel;
use crate::Renderer;
use image::{DynamicImage, RgbaImage};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    projection: [[f32; 4]; 4],
    camera: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    u_slice: f32,
    u_material_num: f32,
    _padding: [f32; 2],
}

pub struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buffer: Option<wgpu::Buffer>,
    vertex_buffer: wgpu::Buffer,
    target_texture: Option<wgpu::Texture>,
    read_buffer: Option<wgpu::Buffer>,
    width: u32,
    height: u32,
}

impl WgpuRenderer {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        let vertex_data: [f32; 18] = [
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0, -1.0, 0.0,
             1.0,  1.0, 0.0,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertex_data),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            device,
            queue,
            pipeline: None,
            bind_group: None,
            uniform_buffer: None,
            vertex_buffer,
            target_texture: None,
            read_buffer: None,
            width: 0,
            height: 0,
        })
    }
}

impl Renderer for WgpuRenderer {
    fn init(&mut self, width: u32, height: u32) -> Result<(), Box<dyn std::error::Error>> {
        self.width = width;
        self.height = height;

        let texture_extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Target Texture"),
            size: texture_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let bytes_per_row = (width * 4 + 255) & !255;
        let output_buffer_size = (bytes_per_row * height) as wgpu::BufferAddress;
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        self.target_texture = Some(texture);
        self.read_buffer = Some(read_buffer);

        Ok(())
    }

    fn prepare(&mut self, model: &IrmfModel) -> Result<(), Box<dyn std::error::Error>> {
        let lang = model.header.language.as_deref().unwrap_or("glsl");
        
        let shader_source = if lang == "wgsl" {
            format!(r#"
struct Uniforms {{
    projection: mat4x4<f32>,
    camera: mat4x4<f32>,
    model: mat4x4<f32>,
    u_slice: f32,
    u_material_num: f32,
}};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {{
    @builtin(position) position: vec4<f32>,
    @location(0) fragVert: vec3<f32>,
}};

@vertex
fn vs_main(@location(0) vert: vec3<f32>) -> VertexOutput {{
    var out: VertexOutput;
    out.position = uniforms.projection * uniforms.camera * uniforms.model * vec4<f32>(vert, 1.0);
    out.fragVert = vert;
    return out;
}}

{}

@fragment
fn fs_main(@location(0) fragVert: vec3<f32>) -> @location(0) vec4<f32> {{
    let m = mainModel4(fragVert);
    var color = 0.0;
    let mat_num = i32(uniforms.u_material_num);
    if mat_num == 1 {{ color = m.x; }}
    else if mat_num == 2 {{ color = m.y; }}
    else if mat_num == 3 {{ color = m.z; }}
    else if mat_num == 4 {{ color = m.w; }}
    return vec4<f32>(color, color, color, 1.0);
}}
"#, model.shader)
        } else {
            return Err("GLSL support in WgpuRenderer not yet implemented".into());
        };

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });

        let uniform_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 3 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
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

        self.pipeline = Some(pipeline);
        self.bind_group = Some(bind_group);
        self.uniform_buffer = Some(uniform_buffer);
        
        Ok(())
    }

    fn render(&mut self, z: f32, material_num: usize) -> Result<DynamicImage, Box<dyn std::error::Error>> {
        let pipeline = self.pipeline.as_ref().ok_or("Pipeline not prepared")?;
        let bind_group = self.bind_group.as_ref().ok_or("Bind group not prepared")?;
        let uniform_buffer = self.uniform_buffer.as_ref().ok_or("Uniform buffer not prepared")?;
        let target_texture = self.target_texture.as_ref().ok_or("Target texture not initialized")?;
        let read_buffer = self.read_buffer.as_ref().ok_or("Read buffer not initialized")?;

        let identity = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let uniforms = Uniforms {
            projection: identity,
            camera: identity,
            model: identity,
            u_slice: z,
            u_material_num: material_num as f32,
            _padding: [0.0, 0.0],
        };
        self.queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

        {
            let render_target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &render_target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..6, 0..1);
        }

        let bytes_per_row = (self.width * 4 + 255) & !255;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: read_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = read_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;

        let data = buffer_slice.get_mapped_range();
        let mut rgba = RgbaImage::new(self.width, self.height);
        for (y, row) in data.chunks_exact(bytes_per_row as usize).take(self.height as usize).enumerate() {
            for (x, pixel) in row.chunks_exact(4).take(self.width as usize).enumerate() {
                rgba.put_pixel(x as u32, y as u32, image::Rgba([pixel[0], pixel[1], pixel[2], pixel[3]]));
            }
        }
        drop(data);
        read_buffer.unmap();

        Ok(DynamicImage::ImageRgba8(rgba))
    }
}