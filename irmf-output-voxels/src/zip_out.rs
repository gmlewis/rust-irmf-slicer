use chrono::{Datelike, Local, Timelike};
use irmf_slicer::{DynamicImage, IrmfResult, Renderer, Slicer};
use std::fs::File;
use std::io::Write;
use zip::write::FileOptions;

pub fn slice_to_zip<R: Renderer, F, P>(
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
    let file = File::create(filename).map_err(|e| anyhow::anyhow!("File::create: {}", e))?;
    let mut zip = zip::ZipWriter::new(file);

    println!("Rendering Z-slices for ZIP...");
    slicer
        .prepare_render_z()
        .map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    let now = Local::now();
    let dt = zip::DateTime::from_date_and_time(
        now.year() as u16,
        now.month() as u8,
        now.day() as u8,
        now.hour() as u8,
        now.minute() as u8,
        now.second() as u8,
    )
    .map_err(|_| anyhow::anyhow!("Invalid current local time for ZIP"))?;

    let options = FileOptions::<()>::default().last_modified_time(dt);

    let total_slices = slicer.num_z_slices();
    slicer
        .render_z_slices(material_num, |z_idx, _z, _radius, img| {
            if let Some(ref mut f) = on_slice {
                f(&img)?;
            }
            if let Some(ref mut p) = on_progress {
                p(z_idx + 1, total_slices);
            }
            let slice_name = format!("{:04}.png", z_idx);
            zip.start_file(slice_name, options)
                .map_err(|e| anyhow::anyhow!("zip.start_file: {}", e))?;

            let mut buffer = Vec::new();
            img.write_to(
                &mut std::io::Cursor::new(&mut buffer),
                image::ImageFormat::Png,
            )
            .map_err(|e| anyhow::anyhow!("img.write_to: {}", e))?;

            zip.write_all(&buffer)
                .map_err(|e| anyhow::anyhow!("zip.write_all: {}", e))?;

            Ok(())
        })
        .map_err(|e| anyhow::anyhow!("render_z_slices: {}", e))?;

    zip.finish().map_err(|e| anyhow::anyhow!("zip.finish: {}", e))?;

    Ok(())
}