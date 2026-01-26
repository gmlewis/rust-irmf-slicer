use crate::{BinVox, Mesh, Triangle};
use irmf_slicer::{IrmfError, IrmfResult};
use wgpu::util::DeviceExt;

/// GPU-based Marching Cubes implementation.
pub struct GpuMarchingCubes {
    pipeline: wgpu::ComputePipeline,
    tri_table_buffer: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTriangle {
    normal: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    v3: [f32; 4],
}

impl GpuMarchingCubes {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Marching Cubes Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("mc.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Marching Cubes Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tri_table_flat: Vec<i32> = crate::TRI_TABLE.iter().flatten().copied().collect();
        let tri_table_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TRI_TABLE Buffer"),
            contents: bytemuck::cast_slice(&tri_table_flat),
            usage: wgpu::BufferUsages::STORAGE,
        });

        Self {
            pipeline,
            tri_table_buffer,
        }
    }

    pub async fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        binvox: &BinVox,
    ) -> IrmfResult<Mesh> {
        device.on_uncaptured_error(Box::new(|error| {
            panic!("GpuMarchingCubes WGPU error: {}", error);
        }));

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);

        let nx = binvox.nx as u32;
        let ny = binvox.ny as u32;
        let nz = binvox.nz as u32;

        let params = Params {
            nx,
            ny,
            nz,
            min_x: binvox.min_x as f32,
            min_y: binvox.min_y as f32,
            min_z: binvox.min_z as f32,
            dx: ((binvox.max_x - binvox.min_x) / (binvox.nx as f64)) as f32,
            dy: ((binvox.max_y - binvox.min_y) / (binvox.ny as f64)) as f32,
            dz: ((binvox.max_z - binvox.min_z) / (binvox.nz as f64)) as f32,
            _padding: [0; 3],
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Voxels are bit-packed, but wgpu wants 4-byte alignment for storage buffers
        // We might need to pad the data.
        let mut voxel_data = binvox.data.clone();
        while voxel_data.len() % 4 != 0 {
            voxel_data.push(0);
        }

        let voxel_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Voxel Buffer"),
            contents: &voxel_data,
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Output buffer: count + array of triangles
        // We need to estimate the maximum number of triangles that can fit in a single
        // GPU buffer based on hardware limits.
        //
        // NOTE: This is a single-pass approach. If a design is extremely complex and
        // exceeds this limit, the compute shader will stop writing triangles,
        // resulting in "holes" in the final STL mesh. 10M triangles is a very high
        // limit for most 3D-printable designs, but if users hit this frequently,
        // a "tiled" approach (processing the voxel grid in smaller blocks)
        // should be implemented to support effectively infinite mesh sizes.
        let tri_size = std::mem::size_of::<GpuTriangle>() as u64;
        let limit = device
            .limits()
            .max_buffer_size
            .min(device.limits().max_storage_buffer_binding_size as u64);
        let max_triangles_from_limit = (limit - 16) / tri_size;

        // Estimate max triangles: each cell can produce up to 5 triangles.
        // We cap this at 10M triangles or the hardware limit.
        let max_triangles = (nx as u64 * ny as u64 * nz as u64)
            .saturating_mul(5)
            .min(max_triangles_from_limit)
            .min(10_000_000) as u32;

        let output_buffer_size = 16 + max_triangles as u64 * tri_size;

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MC Bind Group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.tri_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("MC Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MC Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch enough workgroups to cover (nx+2, ny+2, nz+2)
            // Using workgroup size (8, 8, 2)
            cpass.dispatch_workgroups(
                (nx + 2).div_ceil(8),
                (ny + 2).div_ceil(8),
                (nz + 2).div_ceil(2),
            );
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, output_buffer_size);

        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);

        if let Some(error) = device.pop_error_scope().await {
            return Err(IrmfError::RendererError(format!(
                "GpuMarchingCubes WGPU Out of Memory error: {}",
                error
            )));
        }
        if let Some(error) = device.pop_error_scope().await {
            return Err(IrmfError::RendererError(format!(
                "GpuMarchingCubes WGPU Validation error: {}",
                error
            )));
        }

        let buffer_slice = readback_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            sender.send(v).unwrap();
        });

        device.poll(wgpu::Maintain::Wait);
        receiver
            .await
            .map_err(|_| IrmfError::RendererError("oneshot channel canceled".to_string()))??;

        let data = buffer_slice.get_mapped_range();
        let gpu_count = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if gpu_count >= max_triangles {
            eprintln!(
                "Warning: Maximum triangle limit ({}) reached.",
                max_triangles
            );
            return Err(IrmfError::RendererError(format!(
                "Maximum triangle limit ({}) reached",
                max_triangles
            )));
        }
        let count = gpu_count;

        let gpu_triangles: &[GpuTriangle] = bytemuck::cast_slice(
            &data[16..16 + count as usize * std::mem::size_of::<GpuTriangle>()],
        );

        let triangles = gpu_triangles
            .iter()
            .map(|gt| {
                Triangle::new(
                    [gt.v1[0], gt.v1[1], gt.v1[2]],
                    [gt.v2[0], gt.v2[1], gt.v2[2]],
                    [gt.v3[0], gt.v3[1], gt.v3[2]],
                    [gt.normal[0], gt.normal[1], gt.normal[2]],
                )
            })
            .collect();

        Ok(Mesh { triangles })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BinVox;
    use pollster::block_on;

    #[test]
    fn test_marching_cubes_gpu_single_voxel() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter: Option<wgpu::Adapter> =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));
        let adapter = match adapter {
            Some(a) => a,
            None => {
                println!("Skipping GPU MC test: No suitable adapter found.");
                return;
            }
        };

        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

        let mut b = BinVox::new(2, 2, 2, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        b.set(0, 0, 0);

        let mc = GpuMarchingCubes::new(&device);
        let mesh = block_on(mc.run(&device, &queue, &b)).unwrap();

        assert_eq!(mesh.triangles.len(), 8);

        for tri in &mesh.triangles {
            for v in &[tri.v1, tri.v2, tri.v3] {
                let ok = (v[0] == 0.0 || v[0] == 1.0 || v[0] == 0.5)
                    && (v[1] == 0.0 || v[1] == 1.0 || v[1] == 0.5)
                    && (v[2] == 0.0 || v[2] == 1.0 || v[2] == 0.5);
                assert!(ok, "Vertex {:?} is not at a boundary or midpoint", v);
            }
        }
    }
}
