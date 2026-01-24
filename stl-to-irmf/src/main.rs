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
    let mut mesh = stl_io::read_stl(&mut file).map_err(|e| anyhow::anyhow!("stl read error: {:?}", e))?;

    println!("Voxelizing mesh (placeholders)...");
    // TODO: implement voxelization
    let volume = VoxelVolume::new(
        [args.res, args.res, args.res],
        Vec3::ZERO,
        Vec3::ONE,
    );

    let mut optimizer = Optimizer::new(volume).await?;
    let irmf = optimizer.generate_irmf();
    
    let output_path = args.output.unwrap_or_else(|| args.input.with_extension("irmf"));
    println!("Writing {}...", output_path.display());
    std::fs::write(output_path, irmf)?;

    Ok(())
}
