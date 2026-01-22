//! STL output generation for IRMF models.
//!
//! This crate provides functionality to slice an IRMF model and generate
//! an STL file using a Marching Cubes algorithm based on the generated voxels.
//!
//! For more information about the IRMF format and its capabilities, visit the
//! [official IRMF website](https://irmf.io).

use image::GenericImageView;
use irmf_output_voxels::BinVox;
use irmf_slicer::{DynamicImage, IrmfResult, Renderer, Slicer};
use std::fs::File;
use std::io::BufWriter;

/// Slices the model and saves the result as an STL file.
///
/// This function orchestrates the rendering of Z-slices, populates a voxel
/// grid, runs the Marching Cubes algorithm to generate a mesh, and writes
/// the mesh to a binary STL file.
///
/// # Arguments
///
/// * `slicer` - The IRMF slicer.
/// * `material_num` - The index of the material to slice.
/// * `filename` - The path to the output STL file.
/// * `on_slice` - Optional callback called after each slice is rendered.
/// * `on_progress` - Optional callback for reporting progress.
pub fn slice_to_stl<R: Renderer, F, P>(
    slicer: &mut Slicer<R>,
    material_num: usize,
    filename: &str,
    mut on_slice: Option<F>,
    mut on_progress: Option<P>,
) -> IrmfResult<()>
where
    F: FnMut(&DynamicImage) -> IrmfResult<()>,
    P: FnMut(usize, usize),
{
    let nx = slicer.num_x_slices();
    let ny = slicer.num_y_slices();
    let nz = slicer.num_z_slices();
    let min = slicer.model.header.min;
    let max = slicer.model.header.max;

    let mut model = BinVox::new(nx, ny, nz, min, max);

    println!("Rendering Z-slices for STL...");
    slicer.prepare_render_z()?;

    let total_slices = slicer.num_z_slices();
    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        if let Some(ref mut f) = on_slice {
            f(&img)?;
        }
        if let Some(ref mut p) = on_progress {
            p(z_idx + 1, total_slices);
        }
        for y in 0..img.height() {
            for x in 0..img.width() {
                let pixel = img.get_pixel(x, y);
                if pixel[0] >= 128 {
                    model.set(x as usize, y as usize, z_idx);
                }
            }
        }
        Ok(())
    })?;

    println!("Converting voxels to STL...");
    let mesh = if let Some((device, queue)) = slicer.renderer.wgpu_device_queue() {
        println!("Using GPU-accelerated Marching Cubes...");
        let gpu_mc = irmf_output_voxels::gpu_mc::GpuMarchingCubes::new(device);
        match pollster::block_on(gpu_mc.run(device, queue, &model)) {
            Ok(mesh) => mesh,
            Err(e) => {
                eprintln!(
                    "GPU-accelerated Marching Cubes failed: {}. Falling back to CPU...",
                    e
                );
                model.marching_cubes(on_progress)
            }
        }
    } else {
        model.marching_cubes(on_progress)
    };

    println!("Writing STL file: {}", filename);
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    mesh.save_stl(&mut writer)?;

    Ok(())
}
