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

    /// Number of optimization iterations
    #[arg(short, long, default_value_t = 1000)]
    iterations: usize,

    /// Use greedy box initialization
    #[arg(short, long)]
    greedy: bool,

    /// Use hierarchical octree initialization
    #[arg(long)]
    octree: bool,
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

    if args.greedy {
        println!("Performing greedy box initialization...");
        optimizer.greedy_box_initialize();
        let initial_count = optimizer.generate_irmf().split("val =").count() - 1;
        println!("Greedy pass produced {} primitives.", initial_count);

        if initial_count > args.max_primitives {
            println!("Decimating to {} primitives...", args.max_primitives);
            optimizer.decimate(args.max_primitives);
        }
    } else {
        // Default to octree if no greedy flag, or if octree flag is set
        println!("Performing hierarchical octree initialization...");
        optimizer.octree_initialize(args.max_primitives).await?;
    }

    if args.iterations > 0 {
        println!("Starting optimization...");
        let mut best_error = f32::MAX;
        let mut last_num_prims = 0;
        for i in 0..args.iterations {
            let error = optimizer.run_iteration().await?;
            let num_prims = optimizer.generate_irmf().split("val =").count() - 1;

            if num_prims > last_num_prims {
                println!("Iteration {}: Added primitive. Total: {}", i, num_prims);
                last_num_prims = num_prims;
            }

            if i % 100 == 0 || error < best_error {
                if error < best_error {
                    best_error = error;
                }
                println!(
                    "Iteration {}: error = {}, primitives = {}",
                    i, error, num_prims
                );
            }

            if error < args.error {
                println!("Target error reached!");
                break;
            }
        }
    } else {
        println!("Skipping optimization iterations as requested.");
    }

    let irmf = optimizer.generate_irmf();

    let output_path = args
        .output
        .unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing {}...", output_path.display());
    std::fs::write(output_path, irmf)?;

    Ok(())
}
