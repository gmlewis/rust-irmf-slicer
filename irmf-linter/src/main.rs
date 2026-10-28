use clap::Parser;
use irmf_include_resolver::has_include_directive;
use irmf_slicer::{IrmfModel, Renderer, WgpuRenderer};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Allow #include directives to fetch shader code from the network
    #[arg(long)]
    remote_includes: bool,

    /// Allow GitHub includes that name a branch or tag instead of a commit
    #[arg(long)]
    allow_mutable_includes: bool,

    /// IRMF files to lint
    files: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if cli.allow_mutable_includes && !cli.remote_includes {
        anyhow::bail!("--allow-mutable-includes requires --remote-includes");
    }

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

        // Check spec compliance
        if let Err(e) = IrmfModel::validate_spec_compliance(&data) {
            eprintln!("Spec compliance error in {}: {}", file.display(), e);
            exit_code = 1;
            // We continue anyway to see if it parses
        }

        let mut model = match IrmfModel::new(&data) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error parsing IRMF header in {}: {}", file.display(), e);
                exit_code = 1;
                continue;
            }
        };

        // Resolve includes only when the caller explicitly opts in. TSL is
        // JavaScript, so an #include there would be remote code execution.
        if has_include_directive(&model.shader) {
            let language = model.header.language.as_deref().unwrap_or("glsl");
            if language == "tsl" {
                eprintln!(
                    "Error: {}: TSL bodies must not contain #include; inline the included code",
                    file.display()
                );
                exit_code = 1;
                continue;
            }
            if !cli.remote_includes {
                eprintln!(
                    "Error: {}: shader contains #include but remote includes are disabled; \
                     pass --remote-includes or inline the included code",
                    file.display()
                );
                exit_code = 1;
                continue;
            }
            let options = irmf_include_resolver::IncludeOptions {
                allow_mutable: cli.allow_mutable_includes,
            };
            model.shader =
                match irmf_include_resolver::resolve_includes_with_options(&model.shader, options)
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("Error resolving includes in {}: {}", file.display(), e);
                        exit_code = 1;
                        continue;
                    }
                };
        }

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
