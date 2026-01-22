//! # IRMF Slicer CLI
//!
//! A powerful command-line utility for slicing Infinite Resolution Materials Format (IRMF) models.
//!
//! This tool renders 3D models defined by IRMF shaders into various formats suitable for
//! 3D printing, visualization, and further processing.
//!
//! ## Key Features
//! - **Multiple Output Formats:** Generate STL, BinVox, SVX, and DLP/Photon files.
//! - **Real-time Visualization:** Optional live preview of the slicing process.
//! - **GPU Acceleration:** High-performance rendering using `wgpu`.
//! - **Flexible Resolution:** Customizable slicing resolution in microns.
//!
//! For more information about the IRMF format and its capabilities, visit the
//! [official IRMF website](https://irmf.io).

use clap::Parser;
use image::DynamicImage;
use indicatif::{ProgressBar, ProgressStyle};
#[cfg(feature = "view")]
use irmf_slicer::{IrmfError, IrmfResult};
#[cfg(feature = "view")]
use minifb::{Window, WindowOptions};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Resolution in microns (default is 42.0)
    #[arg(short, long)]
    res: Option<f64>,

    /// Render slicing to window
    #[arg(short, long)]
    view: bool,

    /// Write binvox files, one per material
    #[arg(long)]
    binvox: bool,

    /// Write ChiTuBox .cbddlp files (same as AnyCubic .photon), one per material
    #[arg(long)]
    dlp: bool,

    /// Write stl files, one per material
    #[arg(long)]
    stl: bool,

    /// Write slices to svx voxel files, one per material
    #[arg(long)]
    svx: bool,

    /// Write slices to zip files, one per material
    #[arg(long)]
    zip: bool,

    /// Input IRMF files
    #[arg(required = true)]
    files: Vec<PathBuf>,
}

#[cfg(feature = "view")]
struct Viewer {
    window: Option<Window>,
    buffer: Vec<u32>,
}

#[cfg(feature = "view")]
impl Viewer {
    fn new(view: bool, width: u32, height: u32) -> Self {
        if !view {
            return Self {
                window: None,
                buffer: Vec::new(),
            };
        }
        let window = Window::new(
            "IRMF Slicer",
            width as usize,
            height as usize,
            WindowOptions::default(),
        )
        .unwrap_or_else(|e| {
            panic!("{}", e);
        });
        Self {
            window: Some(window),
            buffer: vec![0; (width * height) as usize],
        }
    }

    fn update(&mut self, img: &DynamicImage) -> IrmfResult<()> {
        if let Some(ref mut window) = self.window {
            let width = img.width() as usize;
            let height = img.height() as usize;
            let rgba = img.to_rgba8();
            for (i, pixel) in rgba.pixels().enumerate() {
                let r = pixel[0] as u32;
                let g = pixel[1] as u32;
                let b = pixel[2] as u32;
                self.buffer[i] = (r << 16) | (g << 8) | b;
            }
            window
                .update_with_buffer(&self.buffer, width, height)
                .map_err(|e| IrmfError::RendererError(format!("Window update error: {}", e)))?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.binvox && !args.dlp && !args.stl && !args.svx && !args.zip {
        println!(
            "-binvox, -dlp, -stl, -svx, or -zip must be supplied to generate output. Testing IRMF shader compilation only."
        );
    }

    let (x_res, y_res, z_res) = match () {
        _ if args.dlp && args.res.is_none() => (47.25, 47.25, 50.0),
        _ if args.zip && args.res.is_none() => (65.0, 60.0, 30.0),
        _ if args.res.is_none() => (42.0, 42.0, 42.0),
        _ => {
            let r = args.res.unwrap();
            (r, r, r)
        }
    };

    println!(
        "Resolution in microns: X: {}, Y: {}, Z: {}",
        x_res, y_res, z_res
    );

    for file_path in args.files {
        if file_path.extension().and_then(|s| s.to_str()) != Some("irmf") {
            println!("Skipping non-IRMF file {:?}", file_path);
            continue;
        }

        println!("Processing IRMF shader {:?}...", file_path);
        let data = tokio::fs::read(&file_path).await?;

        let mut model = irmf_slicer::IrmfModel::new(&data)
            .map_err(|e| anyhow::anyhow!("IrmfModel::new: {}", e))?;

        #[cfg(feature = "remote-includes")]
        {
            println!("Resolving includes for {}...", file_path.display());
            model.shader = irmf_include_resolver::resolve_includes(&model.shader)
                .await
                .map_err(|e| anyhow::anyhow!("resolve_includes: {}", e))?;
        }
        #[cfg(not(feature = "remote-includes"))]
        {
            if model.shader.contains("#include") {
                eprintln!(
                    "Warning: Shader contains #include but 'remote-includes' feature is disabled."
                );
            }
        }

        let base_name = file_path.with_extension("");
        let base_name_str = base_name.to_str().unwrap();

        let renderer = irmf_slicer::WgpuRenderer::new()
            .await
            .map_err(|e| anyhow::anyhow!("WgpuRenderer::new: {}", e))?;
        let mut slicer =
            irmf_slicer::Slicer::new(model, renderer, x_res as f32, y_res as f32, z_res as f32);

        for material_num in 1..=slicer.model.header.materials.len() {
            let material_name = slicer.model.header.materials[material_num - 1].replace(" ", "-");

            slicer
                .prepare_render_z()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            #[cfg(feature = "view")]
            let mut viewer: Option<Viewer> = None;
            let pb = ProgressBar::new(slicer.num_z_slices() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template(
                        "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                    )
                    .unwrap()
                    .progress_chars("#>-"),
            );

            let mut on_slice = |_img: &DynamicImage| {
                #[cfg(feature = "view")]
                {
                    if viewer.is_none() && args.view {
                        viewer = Some(Viewer::new(true, _img.width(), _img.height()));
                    }
                    if let Some(ref mut v) = viewer {
                        v.update(_img)?;
                    }
                }
                #[cfg(not(feature = "view"))]
                {
                    if args.view {
                        static WARNED: std::sync::atomic::AtomicBool =
                            std::sync::atomic::AtomicBool::new(false);
                        if !WARNED.swap(true, std::sync::atomic::Ordering::SeqCst) {
                            eprintln!("Warning: --view requested but 'view' feature is disabled.");
                        }
                    }
                }
                Ok(())
            };

            let on_progress = |pos: usize, _len: usize| {
                pb.set_position(pos as u64);
            };

            #[cfg(feature = "stl")]
            if args.stl {
                let filename = format!(
                    "{}-mat{:02}-{}.stl",
                    base_name_str, material_num, material_name
                );
                irmf_output_stl::slice_to_stl(
                    &mut slicer,
                    material_num,
                    &filename,
                    Some(&mut on_slice),
                    Some(on_progress),
                )
                .map_err(|e| anyhow::anyhow!("slice_to_stl: {}", e))?;
            }
            #[cfg(not(feature = "stl"))]
            if args.stl {
                eprintln!("Warning: --stl requested but 'stl' feature is disabled.");
            }

            #[cfg(feature = "voxels")]
            {
                if args.zip {
                    let filename = format!(
                        "{}-mat{:02}-{}.zip",
                        base_name_str, material_num, material_name
                    );
                    irmf_output_voxels::zip_out::slice_to_zip(
                        &mut slicer,
                        material_num,
                        &filename,
                        Some(&mut on_slice),
                        Some(on_progress),
                    )
                    .map_err(|e| anyhow::anyhow!("slice_to_zip: {}", e))?;
                }

                if args.binvox {
                    let filename = format!(
                        "{}-mat{:02}-{}.binvox",
                        base_name_str, material_num, material_name
                    );
                    irmf_output_voxels::binvox_out::slice_to_binvox(
                        &mut slicer,
                        material_num,
                        &filename,
                        Some(&mut on_slice),
                        Some(on_progress),
                    )
                    .map_err(|e| anyhow::anyhow!("slice_to_binvox: {}", e))?;
                }

                if args.dlp {
                    let filename = format!(
                        "{}-mat{:02}-{}.cbddlp",
                        base_name_str, material_num, material_name
                    );
                    irmf_output_voxels::photon_out::slice_to_photon(
                        &mut slicer,
                        material_num,
                        &filename,
                        z_res as f32,
                        Some(&mut on_slice),
                        Some(on_progress),
                    )
                    .map_err(|e| anyhow::anyhow!("slice_to_photon: {}", e))?;
                }

                if args.svx {
                    let filename = format!(
                        "{}-mat{:02}-{}.svx",
                        base_name_str, material_num, material_name
                    );
                    irmf_output_voxels::svx_out::slice_to_svx(
                        &mut slicer,
                        material_num,
                        &filename,
                        Some(&mut on_slice),
                        Some(on_progress),
                    )
                    .map_err(|e| anyhow::anyhow!("slice_to_svx: {}", e))?;
                }
            }
            #[cfg(not(feature = "voxels"))]
            if args.zip || args.binvox || args.dlp || args.svx {
                eprintln!(
                    "Warning: voxel-based output requested but 'voxels' feature is disabled."
                );
            }

            if !args.binvox && !args.dlp && !args.stl && !args.svx && !args.zip && args.view {
                slicer
                    .render_z_slices(material_num, |idx, _z, _rad, img| {
                        on_slice(&img)?;
                        on_progress(idx + 1, 0);
                        Ok(())
                    })
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
            }
            pb.finish_with_message("done");
        }
    }

    Ok(())
}
