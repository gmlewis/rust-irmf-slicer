use anyhow::Result;
use clap::Parser;
use glam::Vec3;
use std::path::PathBuf;
use volume_to_irmf::{Optimizer, VoxelVolume};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .stl file
    input: PathBuf,

    /// Output .irmf file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Resolution for voxelization
    #[arg(short, long, default_value_t = 128)]
    res: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading {}...", args.input.display());
    let mut file = std::fs::File::open(&args.input)?;
    
    println!("Voxelizing mesh...");
    let volume = VoxelVolume::from_stl(&mut file, [args.res, args.res, args.res])?;

    println!("Dimensions: {}x{}x{}", volume.dims[0], volume.dims[1], volume.dims[2]);

    println!("Initializing optimizer...");
    let mut optimizer = Optimizer::new(volume).await?;

    println!("Starting optimization...");
    let mut best_error = f32::MAX;
    for i in 0..1000 {
        let error = optimizer.run_iteration().await?;
        if error < best_error {
            best_error = error;
            println!("Iteration {}: error = {}", i, error);
        }
    }
    
    let irmf = optimizer.generate_irmf();
    
    let output_path = args.output.unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing {}...", output_path.display());
    std::fs::write(output_path, irmf)?;

    Ok(())
}
