//! SVX (Simple Voxel Format) output generation for IRMF models.

use chrono::{Datelike, Local, Timelike};
use irmf_slicer::{DynamicImage, IrmfError, IrmfResult, Renderer, Slicer};
use std::fs::File;
use std::io::Write;
use zip::write::FileOptions;

/// Slices the model and saves the result as an SVX file.
///
/// An SVX file is a ZIP archive containing a `manifest.xml` file and
/// a series of PNG images representing voxel density.
///
/// # Arguments
///
/// * `slicer` - The IRMF slicer.
/// * `material_num` - The index of the material to slice.
/// * `filename` - The path to the output SVX file.
/// * `on_slice` - Optional callback called after each slice is rendered.
/// * `on_progress` - Optional callback for reporting progress.
pub fn slice_to_svx<R: Renderer, F, P>(
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
    let file = File::create(filename)?;
    let mut zip = zip::ZipWriter::new(file);

    println!("Rendering Z-slices for SVX...");
    slicer.prepare_render_z()?;

    let now = Local::now();
    let dt = zip::DateTime::from_date_and_time(
        now.year() as u16,
        now.month() as u8,
        now.day() as u8,
        now.hour() as u8,
        now.minute() as u8,
        now.second() as u8,
    )
    .map_err(|_| IrmfError::IoError(std::io::Error::other("Invalid current local time for SVX")))?;

    let options = FileOptions::<()>::default().last_modified_time(dt);

    // 1. Write Manifest
    zip.start_file("manifest.xml", options)
        .map_err(|e| IrmfError::IoError(std::io::Error::other(e.to_string())))?;

    let min = slicer.model.header.min;
    let max = slicer.model.header.max;
    let num_z = slicer.num_z_slices();
    let voxel_size_mm = (max[2] - min[2]) / (num_z as f32);
    let voxel_size_m = voxel_size_mm / 1000.0;

    let manifest = format!(
        r#"<?xml version="1.0"?>

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
    let total_slices = num_z;
    slicer.render_z_slices(material_num, |z_idx, _z, _radius, img| {
        if let Some(ref mut f) = on_slice {
            f(&img)?;
        }
        if let Some(ref mut p) = on_progress {
            p(z_idx + 1, total_slices);
        }
        let slice_name = format!("density/slice{:04}.png", z_idx);
        zip.start_file(slice_name, options)
            .map_err(|e| IrmfError::IoError(std::io::Error::other(e.to_string())))?;

        let mut buffer = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageFormat::Png,
        )
        .map_err(|e| IrmfError::IoError(std::io::Error::other(e.to_string())))?;

        zip.write_all(&buffer)?;

        Ok(())
    })?;

    zip.finish()
        .map_err(|e| IrmfError::IoError(std::io::Error::other(e.to_string())))?;

    Ok(())
}
