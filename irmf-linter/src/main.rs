use clap::Parser;
use irmf_include_resolver::resolve_includes;
use irmf_slicer::{IrmfModel, Renderer, WgpuRenderer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// IRMF files to lint
    files: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut exit_code = 0;

    for file in cli.files {
        let data = match std::fs::read(&file) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Error reading {}: {}", file.display(), e);
                exit_code = 1;
                continue;
            }
        };

        let mut model = match IrmfModel::new(&data) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error parsing IRMF header in {}: {}", file.display(), e);
                exit_code = 1;
                continue;
            }
        };

        // Resolve includes
        model.shader = match resolve_includes(&model.shader).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error resolving includes in {}: {}", file.display(), e);
                exit_code = 1;
                continue;
            }
        };

        let mut renderer = match WgpuRenderer::new().await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Error creating WGPU renderer: {}", e);
                exit_code = 1;
                break; // If renderer fails to create, we probably can't continue
            }
        };

        // Initialize with a small size
        if let Err(e) = renderer.init(1, 1) {
            eprintln!("Error initializing renderer: {}", e);
            exit_code = 1;
            continue;
        }

        let left = -1.0;
        let right = 1.0;
        let bottom = -1.0;
        let top = 1.0;
        let vertices = [
            left, bottom, 0.0, right, bottom, 0.0, left, top, 0.0, right, bottom, 0.0, right, top,
            0.0, left, top, 0.0,
        ];
        let projection = glam::Mat4::IDENTITY;
        let camera = glam::Mat4::IDENTITY;
        let model_matrix = glam::Mat4::IDENTITY;
        let vec3_str = "fragVert.xy, u_slice";

        if let Err(e) = renderer.prepare(
            &model,
            &vertices,
            projection,
            camera,
            model_matrix,
            vec3_str,
        ) {
            eprintln!("Lint error in {}: {}", file.display(), e);
            exit_code = 1;
        } else {
            println!("{}: OK", file.display());
        }
    }

    if exit_code != 0 {
        std::process::exit(exit_code);
    }

    Ok(())
}
