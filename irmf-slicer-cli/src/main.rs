use clap::Parser;
use std::path::PathBuf;
use image::DynamicImage;
use minifb::{Window, WindowOptions};
use irmf_slicer::IrmfResult;

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

struct Viewer {
    window: Option<Window>,
    buffer: Vec<u32>,
}

impl Viewer {
    fn new(view: bool, width: u32, height: u32) -> Self {
        if !view {
            return Self { window: None, buffer: Vec::new() };
        }
        let window = Window::new(
            "IRMF Slicer",
            width as usize,
            height as usize,
            WindowOptions::default(),
        ).unwrap_or_else(|e| {
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
            window.update_with_buffer(&self.buffer, width, height)
                .map_err(|e| anyhow::anyhow!("Window update error: {}", e))?;
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.binvox && !args.dlp && !args.stl && !args.svx && !args.zip {
        println!("-binvox, -dlp, -stl, -svx, or -zip must be supplied to generate output. Testing IRMF shader compilation only.");
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

    println!("Resolution in microns: X: {}, Y: {}, Z: {}", x_res, y_res, z_res);

    for file_path in args.files {
        if file_path.extension().and_then(|s| s.to_str()) != Some("irmf") {
            println!("Skipping non-IRMF file {:?}", file_path);
            continue;
        }

        println!("Processing IRMF shader {:?}...", file_path);
        let data = tokio::fs::read(&file_path).await?;
        
        let mut model = irmf_slicer::IrmfModel::new(&data)
            .map_err(|e| anyhow::anyhow!("IrmfModel::new: {}", e))?;
        
        println!("Resolving includes for {}...", file_path.display());
        model.shader = irmf_include_resolver::resolve_includes(&model.shader).await
            .map_err(|e| anyhow::anyhow!("resolve_includes: {}", e))?;

        let base_name = file_path.file_stem().unwrap().to_str().unwrap();

        let renderer = irmf_slicer::WgpuRenderer::new().await
            .map_err(|e| anyhow::anyhow!("WgpuRenderer::new: {}", e))?;
        let mut slicer = irmf_slicer::Slicer::new(model, renderer, x_res as f32, y_res as f32, z_res as f32);

        for material_num in 1..=slicer.model.header.materials.len() {
            let material_name = slicer.model.header.materials[material_num - 1].replace(" ", "-");
            
            // For each material, we might need a window.
            // We'll prepare Z first to get the size.
            slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("{}", e))?;
            let (width, height) = (0, 0); // Need to expose renderer size or calculate it.
            // Actually, let's just let the viewer be created inside the closures.
            
            let mut viewer: Option<Viewer> = None;
            let mut on_slice = |img: &DynamicImage| {
                if viewer.is_none() && args.view {
                    viewer = Some(Viewer::new(true, img.width(), img.height()));
                }
                if let Some(ref mut v) = viewer {
                    v.update(img)?;
                }
                Ok(())
            };

            if args.stl {
                let filename = format!("{}-mat{:02}-{}.stl", base_name, material_num, material_name);
                irmf_output_stl::slice_to_stl(&mut slicer, material_num, &filename, Some(&mut on_slice))
                    .map_err(|e| anyhow::anyhow!("slice_to_stl: {}", e))?;
            }

            if args.zip {
                let filename = format!("{}-mat{:02}-{}.zip", base_name, material_num, material_name);
                irmf_output_voxels::zip_out::slice_to_zip(&mut slicer, material_num, &filename, Some(&mut on_slice))
                    .map_err(|e| anyhow::anyhow!("slice_to_zip: {}", e))?;
            }

            if args.binvox {
                let filename = format!("{}-mat{:02}-{}.binvox", base_name, material_num, material_name);
                irmf_output_voxels::binvox_out::slice_to_binvox(&mut slicer, material_num, &filename, Some(&mut on_slice))
                    .map_err(|e| anyhow::anyhow!("slice_to_binvox: {}", e))?;
            }

            if args.dlp {
                let filename = format!("{}-mat{:02}-{}.cbddlp", base_name, material_num, material_name);
                irmf_output_voxels::photon_out::slice_to_photon(&mut slicer, material_num, &filename, z_res as f32, Some(&mut on_slice))
                    .map_err(|e| anyhow::anyhow!("slice_to_photon: {}", e))?;
            }

            if args.svx {
                let filename = format!("{}-mat{:02}-{}.svx", base_name, material_num, material_name);
                irmf_output_voxels::svx_out::slice_to_svx(&mut slicer, material_num, &filename, Some(&mut on_slice))
                    .map_err(|e| anyhow::anyhow!("slice_to_svx: {}", e))?;
            }
            
            if !args.binvox && !args.dlp && !args.stl && !args.svx && !args.zip {
                // If only testing, we still might want to view
                if args.view {
                    slicer.render_z_slices(material_num, |_idx, _z, _rad, img| {
                        on_slice(&img)
                    }).map_err(|e| anyhow::anyhow!("{}", e))?;
                }
            }
        }
    }

    Ok(())
}