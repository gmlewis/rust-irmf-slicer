use irmf_slicer::{Slicer, Renderer, IrmfResult};
use std::fs::File;
use std::io::BufWriter;
use zip::write::FileOptions;
use image::ImageFormat;

pub fn slice_to_zip<R: Renderer>(slicer: &mut Slicer<R>, material_num: usize, filename: &str) -> IrmfResult<()> {
    let file = File::create(filename).map_err(|e| anyhow::anyhow!("File::create: {}", e))?;
    let mut zip = zip::ZipWriter::new(file);

    println!("Rendering Z-slices for ZIP...");
    slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        let slice_name = format!("{:04}.png", z_idx);
        zip.start_file(slice_name, FileOptions::<()>::default())
            .map_err(|e| anyhow::anyhow!("zip.start_file: {}", e))?;
        
        let mut buffer = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::Png)
            .map_err(|e| anyhow::anyhow!("img.write_to: {}", e))?;
        
        use std::io::Write;
        zip.write_all(&buffer).map_err(|e| anyhow::anyhow!("zip.write_all: {}", e))?;
        
        Ok(())
    }).map_err(|e| anyhow::anyhow!("render_z_slices: {}", e))?;

    zip.finish().map_err(|e| anyhow::anyhow!("zip.finish: {}", e))?;

    Ok(())
}
