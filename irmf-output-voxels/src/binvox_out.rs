//! Binvox output generation for IRMF models.

use crate::BinVox;
use image::GenericImageView;
use irmf_slicer::{DynamicImage, IrmfResult, Renderer, Slicer};
use std::fs::File;
use std::io::BufWriter;

/// Slices the model and saves the result as a Binvox file.
///
/// This function renders Z-slices of the model, populates a voxel grid,
/// and writes it to a file using the Binvox run-length encoding format.
///
/// # Arguments
///
/// * `slicer` - The IRMF slicer.
/// * `material_num` - The index of the material to slice.
/// * `filename` - The path to the output Binvox file.
/// * `on_slice` - Optional callback called after each slice is rendered.
/// * `on_progress` - Optional callback for reporting progress.
pub fn slice_to_binvox<R: Renderer, F, P>(
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

    println!("Rendering Z-slices for Binvox...");
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

    println!("Writing Binvox file: {}", filename);
    let file = File::create(filename)?;
    let mut writer = BufWriter::new(file);
    model.write_binvox(&mut writer)?;

    Ok(())
}
