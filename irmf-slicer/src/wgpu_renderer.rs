use crate::irmf::IrmfModel;
use crate::{IrmfResult, Renderer};
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
    vertex_buffer: Option<wgpu::Buffer>,
    target_texture: Option<wgpu::Texture>,
    read_buffer: Option<wgpu::Buffer>,
    width: u32,
    height: u32,

    // Stored matrices for rendering
    projection: glam::Mat4,
    camera: glam::Mat4,
    model_matrix: glam::Mat4,
}

impl WgpuRenderer {
    pub async fn new() -> IrmfResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok(Self {
            device,
            queue,
            pipeline: None,
            bind_group: None,
            uniform_buffer: None,
            vertex_buffer: None,
            target_texture: None,
            read_buffer: None,
            width: 0,
            height: 0,
            projection: glam::Mat4::IDENTITY,
            camera: glam::Mat4::IDENTITY,
            model_matrix: glam::Mat4::IDENTITY,
        })
    }
}

impl Renderer for WgpuRenderer {
    fn init(&mut self, width: u32, height: u32) -> IrmfResult<()> {
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

    fn prepare(
        &mut self,
        model: &IrmfModel,
        vertices: &[f32],
        projection: glam::Mat4,
        camera: glam::Mat4,
        model_matrix: glam::Mat4,
        vec3_str: &str,
    ) -> IrmfResult<()> {
        self.projection = projection;
        self.camera = camera;
        self.model_matrix = model_matrix;

        // Create vertex buffer with world coordinates
        self.vertex_buffer = Some(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
        );

        let lang = model.header.language.as_deref().unwrap_or("glsl");
        let num_materials = model.header.materials.len();

        if lang == "wgsl" {
            let footer = gen_wgsl_footer(num_materials, vec3_str);
            let shader_source = format!(
                r#"
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

{}
"#,
                model.shader, footer
            );

            let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
            });

            self.create_pipeline(&shader, "vs_main", &shader, "fs_main")
        } else {
            // GLSL support
            let glsl_fs = gen_glsl_full_fs(num_materials, &model.shader, vec3_str);
            let glsl_vs = r#"#version 450
layout(location = 0) in vec3 vert;
layout(location = 0) out vec3 fragVert;
layout(set = 0, binding = 0) uniform Uniforms {
    mat4 projection;
    mat4 camera;
    mat4 model;
    float u_slice;
    float u_material_num;
};
void main() {
    gl_Position = projection * camera * model * vec4(vert, 1.0);
    fragVert = vert;
}
"#;

            // Translate GLSL to WGSL using naga
            let vs_wgsl = translate_glsl_to_wgsl(glsl_vs, naga::ShaderStage::Vertex)?;
            let fs_wgsl = translate_glsl_to_wgsl(&glsl_fs, naga::ShaderStage::Fragment)?;

            let vs_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("VS Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(vs_wgsl)),
            });
            let fs_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("FS Shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(fs_wgsl)),
            });

            self.create_pipeline(&vs_module, "main", &fs_module, "main")
        }
    }

    fn render(&mut self, slice_depth: f32, material_num: usize) -> IrmfResult<DynamicImage> {
        let pipeline = self.pipeline.as_ref().ok_or("Pipeline not prepared")?;
        let bind_group = self.bind_group.as_ref().ok_or("Bind group not prepared")?;
        let uniform_buffer = self.uniform_buffer.as_ref().ok_or("Uniform buffer not prepared")?;
        let target_texture = self
            .target_texture
            .as_ref()
            .ok_or("Target texture not initialized")?;
        let read_buffer = self.read_buffer.as_ref().ok_or("Read buffer not initialized")?;
        let vertex_buffer = self.vertex_buffer.as_ref().ok_or("Vertex buffer not prepared")?;

        let uniforms = Uniforms {
            projection: self.projection.to_cols_array_2d(),
            camera: self.camera.to_cols_array_2d(),
            model: self.model_matrix.to_cols_array_2d(),
            u_slice: slice_depth,
            u_material_num: material_num as f32,
            _padding: [0.0, 0.0],
        };
        self.queue
            .write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

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
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
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
        for (y, row) in data
            .chunks_exact(bytes_per_row as usize)
            .take(self.height as usize)
            .enumerate()
        {
            for (x, pixel) in row.chunks_exact(4).take(self.width as usize).enumerate() {
                rgba.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([pixel[0], pixel[1], pixel[2], pixel[3]]),
                );
            }
        }
        drop(data);
        read_buffer.unmap();

        Ok(DynamicImage::ImageRgba8(rgba))
    }
}

impl WgpuRenderer {
    fn create_pipeline(
        &mut self,
        vs_module: &wgpu::ShaderModule,
        vs_entry: &str,
        fs_module: &wgpu::ShaderModule,
        fs_entry: &str,
    ) -> IrmfResult<()> {
        let uniform_size = std::mem::size_of::<Uniforms>() as wgpu::BufferAddress;
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: vs_module,
                    entry_point: Some(vs_entry),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 3 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: fs_module,
                    entry_point: Some(fs_entry),
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
}

fn translate_glsl_to_wgsl(glsl: &str, stage: naga::ShaderStage) -> IrmfResult<String> {
    let mut parser = naga::front::glsl::Frontend::default();
    let module = parser
        .parse(
            &naga::front::glsl::Options {
                stage,
                defines: rustc_hash::FxHashMap::<String, String>::default(),
            },
            glsl,
        )
        .map_err(|e| format!("GLSL Parse Error: {:?}", e))?;

    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("Naga Validation Error: {:?}", e))?;

    let wgsl = naga::back::wgsl::write_string(&module, &info, naga::back::wgsl::WriterFlags::empty())
        .map_err(|e| format!("WGSL Back-end Error: {:?}", e))?;

    Ok(wgsl)
}

fn gen_glsl_full_fs(num_materials: usize, shader: &str, vec3_str: &str) -> String {
    let call = if num_materials <= 4 {
        format!("vec4 m; mainModel4(m, vec3({}));", vec3_str)
    } else if num_materials <= 9 {
        format!("mat3 m; mainModel9(m, vec3({}));", vec3_str)
    } else {
        format!("mat4 m; mainModel16(m, vec3({}));", vec3_str)
    };

    let cases = if num_materials <= 4 {
        r#"
        case 1: outputColor = vec4(m.x); break;
        case 2: outputColor = vec4(m.y); break;
        case 3: outputColor = vec4(m.z); break;
        case 4: outputColor = vec4(m.w); break;
"#
    } else if num_materials <= 9 {
        r#"
        case 1: outputColor = vec4(m[0][0]); break;
        case 2: outputColor = vec4(m[0][1]); break;
        case 3: outputColor = vec4(m[0][2]); break;
        case 4: outputColor = vec4(m[1][0]); break;
        case 5: outputColor = vec4(m[1][1]); break;
        case 6: outputColor = vec4(m[1][2]); break;
        case 7: outputColor = vec4(m[2][0]); break;
        case 8: outputColor = vec4(m[2][1]); break;
        case 9: outputColor = vec4(m[2][2]); break;
"#
    } else {
        r#"
        case 1: outputColor = vec4(m[0][0]); break;
        case 2: outputColor = vec4(m[0][1]); break;
        case 3: outputColor = vec4(m[0][2]); break;
        case 4: outputColor = vec4(m[0][3]); break;
        case 5: outputColor = vec4(m[1][0]); break;
        case 6: outputColor = vec4(m[1][1]); break;
        case 7: outputColor = vec4(m[1][2]); break;
        case 8: outputColor = vec4(m[1][3]); break;
        case 9: outputColor = vec4(m[2][0]); break;
        case 10: outputColor = vec4(m[2][1]); break;
        case 11: outputColor = vec4(m[2][2]); break;
        case 12: outputColor = vec4(m[2][3]); break;
        case 13: outputColor = vec4(m[3][0]); break;
        case 14: outputColor = vec4(m[3][1]); break;
        case 15: outputColor = vec4(m[3][2]); break;
        case 16: outputColor = vec4(m[3][3]); break;
"#
    };

    format!(
        r#"#version 450
precision highp float;
layout(location = 0) in vec3 fragVert;
layout(location = 0) out vec4 outputColor;
layout(set = 0, binding = 0) uniform Uniforms {{
    mat4 projection;
    mat4 camera;
    mat4 model;
    float u_slice;
    float u_material_num;
}};

{}

void main() {{
    float u_slice = u_slice;
    {}
    int mat_num = int(u_material_num);
    switch(mat_num) {{
        {}
        default: outputColor = vec4(0.0); break;
    }}
}}
"#,
        shader, call, cases
    )
}

fn gen_wgsl_footer(num_materials: usize, vec3_str: &str) -> String {
    let call = if num_materials <= 4 {
        format!("let m = mainModel4(vec3<f32>({}));", vec3_str)
    } else if num_materials <= 9 {
        format!("let m = mainModel9(vec3<f32>({}));", vec3_str)
    } else {
        format!("let m = mainModel16(vec3<f32>({}));", vec3_str)
    };

    let cases = if num_materials <= 4 {
        r#"
        case 1: { color = m.x; }
        case 2: { color = m.y; }
        case 3: { color = m.z; }
        case 4: { color = m.w; }
"#
    } else if num_materials <= 9 {
        r#"
        case 1: { color = m[0][0]; }
        case 2: { color = m[0][1]; }
        case 3: { color = m[0][2]; }
        case 4: { color = m[1][0]; }
        case 5: { color = m[1][1]; }
        case 6: { color = m[1][2]; }
        case 7: { color = m[2][0]; }
        case 8: { color = m[2][1]; }
        case 9: { color = m[2][2]; }
"#
    } else {
        r#"
        case 1: { color = m[0][0]; }
        case 2: { color = m[0][1]; }
        case 3: { color = m[0][2]; }
        case 4: { color = m[0][3]; }
        case 5: { color = m[1][0]; }
        case 6: { color = m[1][1]; }
        case 7: { color = m[1][2]; }
        case 8: { color = m[1][3]; }
        case 9: { color = m[2][0]; }
        case 10: { color = m[2][1]; }
        case 11: { color = m[2][2]; }
        case 12: { color = m[2][3]; }
        case 13: { color = m[3][0]; }
        case 14: { color = m[3][1]; }
        case 15: { color = m[3][2]; }
        case 16: { color = m[3][3]; }
"#
    };

    format!(
        r#"
@fragment
fn fs_main(@location(0) fragVert: vec3<f32>) -> @location(0) vec4<f32> {{
    let u_slice = uniforms.u_slice;
    {}
    var color = 0.0;
    let mat_num = i32(uniforms.u_material_num);
    switch mat_num {{
        {}
        default: {{ color = 0.0; }}
    }}
    return vec4<f32>(color, color, color, 1.0);
}}
"#,
        call, cases
    )
}