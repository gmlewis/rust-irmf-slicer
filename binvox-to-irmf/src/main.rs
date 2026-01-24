use anyhow::Result;
use clap::Parser;
use glam::Vec3;
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
    let mut file = std::fs::File::open(&args.input)?;
    let model = binvox::read(&mut file).map_err(|e| anyhow::anyhow!("binvox read error: {:?}", e))?;

    let dims = [model.width as u32, model.height as u32, model.depth as u32];
    println!("Dimensions: {}x{}x{}", dims[0], dims[1], dims[2]);

    // binvox coordinates: x is fastest, then z, then y?
    // Actually binvox::Model.data is a Vec<bool>
    // The binvox crate says: "The order of voxels is (x, z, y), where x is the inner-most loop."
    
    let mut volume = VoxelVolume::new(
        dims,
        Vec3::new(model.tx, model.ty, model.tz),
        Vec3::new(
            model.tx + model.scale,
            model.ty + model.scale,
            model.tz + model.scale,
        ),
    );

    for (i, &filled) in model.data.iter().enumerate() {
        if filled {
            let x = i as u32 % dims[0];
            let z = (i as u32 / dims[0]) % dims[2];
            let y = i as u32 / (dims[0] * dims[2]);
            volume.set(x, y, z, 1.0);
        }
    }

    println!("Initializing optimizer...");
    let mut optimizer = Optimizer::new(volume).await?;

    println!("Starting optimization (placeholders)...");
    // TODO: loop and run iterations
    
    let irmf = optimizer.generate_irmf();
    
    let output_path = args.output.unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing {}...", output_path.display());
    std::fs::write(output_path, irmf)?;

    Ok(())
}
