//! Converts BinVox voxel files to optimized IRMF shaders.
//!
//! This tool reads a BinVox file, applies lossless cuboid merging optimization,
//! and generates an IRMF shader that efficiently represents the 3D model.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use volume_to_irmf::{Optimizer, VoxelVolume};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .binvox file
    input: PathBuf,

    /// Output .irmf file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// IRMF shader language (glsl or wgsl)
    #[arg(short, long)]
    language: Option<String>,

    /// Save intermediate Pass 2 (X-runs) debug IRMF
    #[arg(long)]
    pass2: Option<PathBuf>,

    /// Save intermediate Pass 3 (XY-planes) debug IRMF
    #[arg(long)]
    pass3: Option<PathBuf>,

    /// Use GPU for optimization
    #[arg(long)]
    gpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading {} ...", args.input.display());
    let file = std::fs::File::open(&args.input)?;
    let volume = VoxelVolume::from_binvox(file)?;

    println!(
        "Dimensions: {}x{}x{}",
        volume.dims[0], volume.dims[1], volume.dims[2]
    );

    println!("Initializing lossless optimizer...");
    let mut optimizer = Optimizer::new(volume, args.gpu).await?;

    println!("Running lossless cuboid merging algorithm...");
    optimizer.run_lossless().await?;

    if let Some(path) = args.pass2 {
        println!("Writing Pass 2 debug IRMF to {} ...", path.display());
        std::fs::write(path, optimizer.generate_pass2_irmf())?;
    }

    if let Some(path) = args.pass3 {
        println!("Writing Pass 3 debug IRMF to {} ...", path.display());
        std::fs::write(path, optimizer.generate_pass3_irmf())?;
    }

    let language = args.language.unwrap_or_else(|| "glsl".to_string());
    let irmf = optimizer.generate_irmf(language);
    let output_path = args
        .output
        .unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing final IRMF to {} ...", output_path.display());
    std::fs::write(output_path, irmf)?;

    let stats = &optimizer.stats;
    println!(
        "Done in {:?}. Produced {} cuboids.",
        stats.duration,
        optimizer.cuboid_count()
    );

    Ok(())
}
