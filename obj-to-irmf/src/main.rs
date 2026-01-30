//! Converts Wavefront OBJ mesh files to optimized IRMF shaders.
//!
//! This tool voxelizes an OBJ mesh, applies lossless cuboid merging optimization,
//! and generates an IRMF shader that efficiently represents the 3D model.

use anyhow::Result;
use clap::Parser;
use glam::Vec3;
use std::path::PathBuf;
use volume_to_irmf::{Optimizer, VoxelVolume};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input .obj file
    input: PathBuf,

    /// Output .irmf file
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// IRMF shader language (glsl or wgsl)
    #[arg(short, long)]
    language: Option<String>,

    /// Resolution for voxelization
    #[arg(short, long, default_value_t = 64)]
    res: u32,

    /// Save intermediate Pass 2 (X-runs) debug IRMF
    #[arg(long)]
    pass2: Option<PathBuf>,

    /// Save intermediate Pass 3 (XY-planes) debug IRMF
    #[arg(long)]
    pass3: Option<PathBuf>,

    /// Dump debug information to stdout
    #[arg(long)]
    debug: bool,

    /// Use GPU for voxelization and optimization
    #[arg(long)]
    gpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Reading {} ...", args.input.display());
    let (models, _materials) = tobj::load_obj(
        &args.input,
        &tobj::LoadOptions {
            triangulate: true,
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("obj load error: {:?}", e))?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for model in models {
        let mesh = &model.mesh;
        let vertex_offset = vertices.len() as u32;

        for i in 0..mesh.positions.len() / 3 {
            let p = Vec3::new(
                mesh.positions[3 * i],
                mesh.positions[3 * i + 1],
                mesh.positions[3 * i + 2],
            );
            min = min.min(p);
            max = max.max(p);
            vertices.push(p);
        }

        for &idx in &mesh.indices {
            indices.push(vertex_offset + idx);
        }
    }

    println!(
        "Mesh has {} vertices and {} faces.",
        vertices.len(),
        indices.len() / 3
    );

    let size = max - min;
    min -= size * 0.05;
    max += size * 0.05;

    let volume = if args.gpu {
        println!("Voxelizing mesh on GPU at resolution {} ...", args.res);
        VoxelVolume::gpu_voxelize(vertices, indices, [args.res, args.res, args.res], min, max)
            .await?
    } else {
        println!("Voxelizing mesh on CPU at resolution {} ...", args.res);
        VoxelVolume::cpu_voxelize(vertices, indices, [args.res, args.res, args.res], min, max)?
    };

    println!(
        "Dimensions: {}x{}x{}",
        volume.dims[0], volume.dims[1], volume.dims[2]
    );

    if args.debug {
        println!("\nVoxels after voxelizing model");
        let mut i = 0;
        for z in 0..volume.dims[2] {
            for y in 0..volume.dims[1] {
                for x in 0..volume.dims[0] {
                    if volume.get(x, y, z) > 0.5 {
                        println!("{}: ({},{},{})", i, x, y, z);
                        i += 1;
                    }
                }
            }
        }
    }

    println!("Initializing lossless optimizer...");
    let mut optimizer = Optimizer::new(volume, args.gpu).await?;

    println!("Running lossless cuboid merging algorithm...");
    optimizer.run_lossless().await?;

    if args.debug {
        println!("\nCuboids after pass 2");
        for (i, res) in optimizer.pass2_results.iter().enumerate() {
            println!(
                "{}: ({},{},{})-({},{},{})",
                i, res[0], res[2], res[3], res[1], res[2], res[3]
            );
        }

        println!("\nCuboids after pass 3");
        for (i, (rect, z)) in optimizer.pass3_results.iter().enumerate() {
            println!(
                "{}: ({},{},{})-({},{},{})",
                i, rect[0], rect[2], z, rect[1], rect[3], z
            );
        }

        println!("\nCuboids after pass 4");
        for (i, c) in optimizer.cuboids.iter().enumerate() {
            println!(
                "{}: ({},{},{})-({},{},{})",
                i, c.x1, c.y1, c.z1, c.x2, c.y2, c.z2
            );
        }
    }

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
