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

    /// Maximum number of primitives
    #[arg(short, long, default_value_t = 100)]
    max_primitives: usize,

    /// Target error
    #[arg(short, long, default_value_t = 0.01)]
    error: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading {}...", args.input.display());
    let file = std::fs::File::open(&args.input)?;
    let volume = VoxelVolume::from_binvox(file)?;

    println!(
        "Dimensions: {}x{}x{}",
        volume.dims[0], volume.dims[1], volume.dims[2]
    );

    println!("Initializing optimizer...");
    let mut optimizer = Optimizer::new(volume).await?;

    println!("Starting optimization...");
    let mut best_error = f32::MAX;
    for i in 0..1000 {
        // 1000 iterations for testing
        let error = optimizer.run_iteration().await?;
        if error < best_error {
            best_error = error;
            println!("Iteration {}: error = {}", i, error);
        }
        if error < args.error {
            println!("Target error reached!");
            break;
        }
    }

    let irmf = optimizer.generate_irmf();

    let output_path = args
        .output
        .unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing {}...", output_path.display());
    std::fs::write(output_path, irmf)?;

    Ok(())
}
