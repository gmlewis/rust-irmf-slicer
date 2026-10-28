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
//! ## Usage
//!
//! ### Installation
//!
//! After you have a recent version of [Rust](https://rustup.rs/) installed,
//! run the following command in a terminal window:
//!
//! ```sh
//! $ cargo install --path irmf-slicer-cli
//! ```
//!
//! (Or `$ cargo install irmf-slicer-cli` if you are installing from [crates.io](https://crates.io).)
//!
//! ### Examples
//!
//! To slice one or more `.irmf` files, just list them on the command line.
//!
//! Slice a model into STL files:
//! ```sh
//! irmf-slicer-cli --stl examples/001-sphere/sphere-1.irmf
//! ```
//!
//! Slice a model for a resin printer (DLP):
//! ```sh
//! irmf-slicer-cli --dlp examples/001-sphere/sphere-1.irmf
//! ```
//!
//! View the slicing process in real-time:
//! ```sh
//! irmf-slicer-cli --view --zip examples/002-cube/cube-1.irmf
//! ```
//!
//! The output files will be saved in the same directory as the original
//! input IRMF files.
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

    /// Path to irmf-tsl-compile (default: $IRMF_TSL_COMPILER or PATH)
    #[arg(long)]
    tsl_compiler: Option<String>,

    /// TSL compilation cache directory (default: $IRMF_TSL_CACHE_DIR)
    #[arg(long)]
    tsl_cache_dir: Option<String>,

    /// Allow #include directives to fetch shader code from lygia.xyz or
    /// raw.githubusercontent.com. Remote includes are executable code fetched
    /// at load time, so only use this with models you trust.
    #[arg(long)]
    remote_includes: bool,

    /// Allow GitHub includes that name a branch or tag instead of a commit.
    /// Requires --remote-includes and is less reproducible.
    #[arg(long)]
    allow_mutable_includes: bool,

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

    if args.allow_mutable_includes && !args.remote_includes {
        anyhow::bail!("--allow-mutable-includes requires --remote-includes");
    }

    if let Some(ref path) = args.tsl_compiler {
        irmf_slicer::tsl_compiler::set_tsl_compiler(path.clone());
    }
    if let Some(ref dir) = args.tsl_cache_dir {
        irmf_slicer::tsl_compiler::set_tsl_cache_dir(dir.clone());
    }

    if !args.binvox && !args.dlp && !args.stl && !args.svx && !args.zip {
        println!("-binvox, -dlp, -stl, -svx, or -zip not supplied: compiling and validating only.");
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

        if irmf_include_resolver::has_include_directive(&model.shader) {
            let language = model.header.language.as_deref().unwrap_or("glsl");
            if language == "tsl" {
                anyhow::bail!(
                    "{}: TSL bodies must not contain #include; inline the included code",
                    file_path.display()
                );
            }
            if !args.remote_includes {
                anyhow::bail!(
                    "{}: shader contains #include but remote includes are disabled; \
                     pass --remote-includes to fetch them, or inline the included code",
                    file_path.display()
                );
            }
            if args.allow_mutable_includes {
                eprintln!(
                    "WARNING: --allow-mutable-includes accepts branch/tag includes, \
                     which can change after review."
                );
            }
            #[cfg(feature = "remote-includes")]
            {
                println!("Resolving includes for {}...", file_path.display());
                let options = irmf_include_resolver::IncludeOptions {
                    allow_mutable: args.allow_mutable_includes,
                };
                model.shader =
                    irmf_include_resolver::resolve_includes_with_options(&model.shader, options)
                        .await
                        .map_err(|e| anyhow::anyhow!("resolve_includes: {}", e))?;
            }
            #[cfg(not(feature = "remote-includes"))]
            {
                anyhow::bail!(
                    "{}: this build was compiled without the 'remote-includes' feature",
                    file_path.display()
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
            let material_name =
                safe_material_name(&slicer.model.header.materials[material_num - 1]);

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

/// Converts an untrusted material name into a safe filename component.
///
/// Material names come from the IRMF header. Without sanitizing, a name
/// containing "/" or ".." would change the output path, and control
/// characters could spoof terminal output or the filesystem.
fn safe_material_name(name: &str) -> String {
    // Keep printable Unicode (letters, digits, combining marks, CJK,
    // punctuation) so material names in any language remain usable, but never
    // path separators, control characters, or characters invalid on common
    // filesystems.
    let mut out = String::new();
    let mut last_dash = false;
    for c in name.chars() {
        let safe = !c.is_control()
            && !c.is_whitespace()
            && c != '\0'
            && !matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|');
        if safe {
            out.push(c);
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    out = out.trim_matches(['.', '-', '_']).to_string();
    if out.is_empty() {
        out.push_str("material");
    }
    // Cap by characters (never splitting a UTF-8 sequence) and by bytes.
    if out.chars().count() > 64 {
        out = out.chars().take(64).collect();
    }
    while out.len() > 200 {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::safe_material_name;

    #[test]
    fn sanitizes_material_names() {
        assert_eq!(safe_material_name("AISI 1018 steel"), "AISI-1018-steel");
        assert_eq!(
            safe_material_name("../../.ssh/authorized_keys"),
            "ssh-authorized_keys"
        );
        assert_eq!(safe_material_name("PLA"), "PLA");
        assert_eq!(safe_material_name(""), "material");
        assert_eq!(safe_material_name(".."), "material");
        assert_eq!(safe_material_name(&"x".repeat(200)).len(), 64);
        assert!(!safe_material_name("a/b").contains('/'));
        assert!(!safe_material_name("a\nb").contains('\n'));
        // Unicode is preserved.
        assert_eq!(safe_material_name("日本語 PLA"), "日本語-PLA");
        assert_eq!(safe_material_name("AISI 1018 钢"), "AISI-1018-钢");
        assert_eq!(safe_material_name("naïve"), "naïve");
        assert_eq!(safe_material_name("PLA·金属"), "PLA·金属");
        for input in ["日".repeat(100), "😀".repeat(100)] {
            let out = safe_material_name(&input);
            assert!(out.len() <= 200);
            assert!(std::str::from_utf8(out.as_bytes()).is_ok());
        }
    }
}
