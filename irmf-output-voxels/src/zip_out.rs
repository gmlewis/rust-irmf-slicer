use irmf_slicer::{Slicer, Renderer, IrmfResult};
use std::fs::File;
use zip::write::FileOptions;
use image::{ImageFormat, DynamicImage};
use chrono::{Datelike, Timelike, Local};

pub fn slice_to_zip<R: Renderer, F>(
    slicer: &mut Slicer<R>, 
    material_num: usize, 
    filename: &str,
    mut on_slice: Option<F>
) -> IrmfResult<()> 
where 
    F: FnMut(&DynamicImage) -> IrmfResult<()>
{
    let file = File::create(filename).map_err(|e| anyhow::anyhow!("File::create: {}", e))?;
    let mut zip = zip::ZipWriter::new(file);

    println!("Rendering Z-slices for ZIP...");
    slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    let now = Local::now();
    let dt = zip::DateTime::from_date_and_time(
        now.year() as u16,
        now.month() as u8,
        now.day() as u8,
        now.hour() as u8,
        now.minute() as u8,
        now.second() as u8,
    ).map_err(|_| anyhow::anyhow!("Invalid current local time for ZIP"))?;
    
    let options = FileOptions::<()>::default().last_modified_time(dt);

    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        if let Some(ref mut f) = on_slice {
            f(&img)?;
        }
        let slice_name = format!("{:04}.png", z_idx);
        zip.start_file(slice_name, options)
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