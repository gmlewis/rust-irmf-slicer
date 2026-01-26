use glam::Vec3;
use volume_to_irmf::{Optimizer, VoxelVolume};

#[tokio::test]
async fn test_optimizer_lossless_cube() {
    let dims = [32, 32, 32];
    let mut volume = VoxelVolume::new(dims, Vec3::ZERO, Vec3::ONE);

    // Create a target cube from (8,8,8) to (24,24,24)
    for z in 8..24 {
        for y in 8..24 {
            for x in 8..24 {
                volume.set(x, y, z, 1.0);
            }
        }
    }

    let mut optimizer = Optimizer::new(volume).await.unwrap();

    println!("Running lossless optimizer...");
    optimizer
        .run_lossless()
        .await
        .expect("Lossless pass failed");

    let irmf = optimizer.generate_irmf("wgsl".to_string());

    // For a single solid cube, the lossless algorithm should ideally produce exactly 1 cuboid.
    // However, depending on how it's implemented, it might produce more if merging is not perfectly greedy across all axes.
    // In our case, Pass 2 will merge X, Pass 3 will merge Y, Pass 4 will merge Z.
    // So for a single cube, it SHOULD produce exactly 1 cuboid.
    let cuboid_count = optimizer.cuboid_count();
    println!("Produced {} cuboids.", cuboid_count);
    assert!(cuboid_count == 1, "Should produce exactly one cuboid");
    assert!(irmf.contains("mainModel4"));
}

#[tokio::test]

async fn test_optimizer_empty_volume() {
    let dims = [16, 16, 16];
    let volume = VoxelVolume::new(dims, Vec3::ZERO, Vec3::ONE);
    let mut optimizer = Optimizer::new(volume).await.unwrap();
    optimizer
        .run_lossless()
        .await
        .expect("Lossless pass failed on empty volume");

    let irmf = optimizer.generate_irmf();

    let cuboid_count = irmf.split("if (xyzRangeCuboid(").count() - 1;
    assert_eq!(cuboid_count, 0, "Empty volume should produce 0 cuboids");
}
