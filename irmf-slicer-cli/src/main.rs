use clap::Parser;
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

        println!("Model: {} by {}", model.header.title.as_deref().unwrap_or("Untitled"), model.header.author.as_deref().unwrap_or("Unknown"));
        println!("Materials: {:?}", model.header.materials);
        println!("Units: {}", model.header.units);
        println!("MBB: {:?} to {:?}", model.header.min, model.header.max);

        let renderer = irmf_slicer::WgpuRenderer::new().await
            .map_err(|e| anyhow::anyhow!("WgpuRenderer::new: {}", e))?;
        let mut slicer = irmf_slicer::Slicer::new(model, renderer, x_res as f32, y_res as f32, z_res as f32);

        println!("Preparing render for Z axis...");
        slicer.prepare_render_z()
            .map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

        let num_slices = slicer.num_z_slices();
        println!("Slicing {} Z-slices...", num_slices);

        for material_num in 1..=slicer.model.header.materials.len() {
            println!("Slicing material {}...", material_num);
            if num_slices > 0 {
                let img = slicer.render_z_slice(0, material_num)
                    .map_err(|e| anyhow::anyhow!("render_z_slice(0): {}", e))?;
                println!("Slice 0 rendered: {}x{}", img.width(), img.height());
                if num_slices > 1 {
                    let img = slicer.render_z_slice(num_slices - 1, material_num)
                        .map_err(|e| anyhow::anyhow!("render_z_slice({}): {}", num_slices - 1, e))?;
                    println!("Slice {} rendered: {}x{}", num_slices - 1, img.width(), img.height());
                }
            }
        }
    }

    Ok(())
}
