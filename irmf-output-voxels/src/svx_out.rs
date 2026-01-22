use irmf_slicer::{Slicer, Renderer, IrmfResult};
use std::fs::File;
use zip::write::FileOptions;
use image::{ImageFormat, DynamicImage};
use chrono::{Datelike, Timelike, Local};
use std::io::Write;

pub fn slice_to_svx<R: Renderer, F>(
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

    println!("Rendering Z-slices for SVX...");
    slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    let now = Local::now();
    let dt = zip::DateTime::from_date_and_time(
        now.year() as u16,
        now.month() as u8,
        now.day() as u8,
        now.hour() as u8,
        now.minute() as u8,
        now.second() as u8,
    ).map_err(|_| anyhow::anyhow!("Invalid current local time for SVX"))?;
    
    let options = FileOptions::<()>::default().last_modified_time(dt);

    // 1. Write Manifest
    zip.start_file("manifest.xml", options)
        .map_err(|e| anyhow::anyhow!("zip.start_file(manifest): {}", e))?;
    
    let min = slicer.model.header.min;
    let max = slicer.model.header.max;
    let num_z = slicer.num_z_slices();
    let voxel_size_mm = (max[2] - min[2]) / (num_z as f32);
    let voxel_size_m = voxel_size_mm / 1000.0;

    let manifest = format!(r#"<?xml version="1.0"?>

<grid version="1.0" gridSizeX="{}" gridSizeY="{}" gridSizeZ="{}"
   voxelSize="{:.6}" subvoxelBits="8" slicesOrientation="Z" >

    <channels>
        <channel type="DENSITY" bits="8" slices="density/slice%04d.png" />
    </channels>

    <materials>
        <material id="1" urn="urn:shapeways:materials/1" />
    </materials>

    <metadata>
        <entry key="author" value="{}" />
        <entry key="creationDate" value="{}" />
    </metadata>
</grid>"#,
        slicer.num_x_slices(),
        slicer.num_y_slices(),
        num_z,
        voxel_size_m,
        slicer.model.header.author.as_deref().unwrap_or("Unknown"),
        slicer.model.header.date.as_deref().unwrap_or("")
    );
    zip.write_all(manifest.as_bytes())?;

    // 2. Write Slices
    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        if let Some(ref mut f) = on_slice {
            f(&img)?;
        }
        let slice_name = format!("density/slice{:04}.png", z_idx);
        zip.start_file(slice_name, options)
            .map_err(|e| anyhow::anyhow!("zip.start_file: {}", e))?;
        
        let mut buffer = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buffer), ImageFormat::Png)
            .map_err(|e| anyhow::anyhow!("img.write_to: {}", e))?;
        
        zip.write_all(&buffer).map_err(|e| anyhow::anyhow!("zip.write_all: {}", e))?;
        
        Ok(())
    }).map_err(|e| anyhow::anyhow!("render_z_slices: {}", e))?;

    zip.finish().map_err(|e| anyhow::anyhow!("zip.finish: {}", e))?;

    Ok(())
}