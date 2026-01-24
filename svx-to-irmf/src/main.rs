use anyhow::Result;
use clap::Parser;
use glam::Vec3;
use std::io::Read;
use std::path::PathBuf;
use volume_to_irmf::{Optimizer, VoxelVolume};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .svx file
    input: PathBuf,

    /// Output .irmf file
    #[arg(short, long)]
    output: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading {}...", args.input.display());
    let file = std::fs::File::open(&args.input)?;
    let mut archive = zip::ZipArchive::new(file)?;

    let mut slices = Vec::new();
    let mut file_names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
    file_names.sort();

    for name in file_names {
        if name.ends_with(".png") || name.ends_with(".jpg") || name.ends_with(".bmp") {
            let mut file = archive.by_name(&name)?;
            let mut buf = Vec::new();
            file.read_to_end(&mut buf)?;
            let img = image::load_from_memory(&buf)?;
            slices.push(img);
        }
    }

    if slices.is_empty() {
        anyhow::bail!("No image slices found in svx");
    }

    println!("Loaded {} slices", slices.len());
    let volume = VoxelVolume::from_slices(slices, Vec3::ZERO, Vec3::ONE)?;

    println!("Initializing optimizer...");
    let mut optimizer = Optimizer::new(volume).await?;

    println!("Starting optimization...");
    for _ in 0..100 { // Fewer iterations for quick test
        optimizer.run_iteration().await?;
    }
    
    let irmf = optimizer.generate_irmf();
    let output_path = args.output.unwrap_or_else(|| args.input.with_extension("irmf"));
    std::fs::write(output_path, irmf)?;

    Ok(())
}
