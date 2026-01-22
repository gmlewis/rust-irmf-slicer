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

        let base_name = file_path.file_stem().unwrap().to_str().unwrap();

        let renderer = irmf_slicer::WgpuRenderer::new().await
            .map_err(|e| anyhow::anyhow!("WgpuRenderer::new: {}", e))?;
        let mut slicer = irmf_slicer::Slicer::new(model, renderer, x_res as f32, y_res as f32, z_res as f32);

        for material_num in 1..=slicer.model.header.materials.len() {
            let material_name = slicer.model.header.materials[material_num - 1].replace(" ", "-");
            
            if args.stl {
                let filename = format!("{}-mat{:02}-{}.stl", base_name, material_num, material_name);
                irmf_output_stl::slice_to_stl(&mut slicer, material_num, &filename)
                    .map_err(|e| anyhow::anyhow!("slice_to_stl: {}", e))?;
            }

            if args.zip {
                let filename = format!("{}-mat{:02}-{}.zip", base_name, material_num, material_name);
                irmf_output_voxels::zip_out::slice_to_zip(&mut slicer, material_num, &filename)
                    .map_err(|e| anyhow::anyhow!("slice_to_zip: {}", e))?;
            }

            if args.binvox {
                let filename = format!("{}-mat{:02}-{}.binvox", base_name, material_num, material_name);
                irmf_output_voxels::binvox_out::slice_to_binvox(&mut slicer, material_num, &filename)
                    .map_err(|e| anyhow::anyhow!("slice_to_binvox: {}", e))?;
            }
            
            // TODO: dlp, svx
        }
    }

    Ok(())
}