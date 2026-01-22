use irmf_output_voxels::{BinVox};
use irmf_slicer::{Slicer, Renderer, IrmfResult};
use std::fs::File;
use std::io::BufWriter;
use image::GenericImageView;

pub fn slice_to_stl<R: Renderer>(slicer: &mut Slicer<R>, material_num: usize, filename: &str) -> IrmfResult<()> {
    let nx = slicer.num_x_slices();
    let ny = slicer.num_y_slices();
    let nz = slicer.num_z_slices();
    let min = slicer.model.header.min;
    let max = slicer.model.header.max;

    let mut model = BinVox::new(nx, ny, nz, min, max);

    println!("Rendering Z-slices for STL...");
    slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        for y in 0..img.height() {
            for x in 0..img.width() {
                let pixel = img.get_pixel(x, y);
                if pixel[0] >= 128 {
                    model.set(x as usize, y as usize, z_idx);
                }
            }
        }
        Ok(())
    }).map_err(|e| anyhow::anyhow!("render_z_slices: {}", e))?;

    println!("Converting voxels to STL...");
    let mesh = model.marching_cubes();
    
    println!("Writing STL file: {}", filename);
    let file = File::create(filename).map_err(|e| anyhow::anyhow!("File::create: {}", e))?;
    let mut writer = BufWriter::new(file);
    mesh.save_stl(&mut writer).map_err(|e| anyhow::anyhow!("Mesh::save_stl: {}", e))?;

    Ok(())
}
