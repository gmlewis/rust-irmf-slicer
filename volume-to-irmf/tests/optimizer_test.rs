use glam::Vec3;
use volume_to_irmf::{BooleanOp, Optimizer, Primitive, VoxelVolume};

#[tokio::test]
async fn test_optimizer_sphere() {
    let dims = [32, 32, 32];
    let mut volume = VoxelVolume::new(dims, Vec3::ZERO, Vec3::ONE);

    // Create a target sphere at [0.5, 0.5, 0.5] with radius 0.25
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let p = Vec3::new(
                    x as f32 / dims[0] as f32,
                    y as f32 / dims[1] as f32,
                    z as f32 / dims[2] as f32,
                );
                if p.distance(Vec3::splat(0.5)) <= 0.25 {
                    volume.set(x, y, z, 1.0);
                }
            }
        }
    }

    let mut optimizer = Optimizer::new(volume).await.unwrap();

    // Add a rough initial sphere
    optimizer.add_primitive(Primitive::new_sphere(
        Vec3::splat(0.4),
        0.2,
        BooleanOp::Union,
    ));

    let mut last_error = 2.0;
    for i in 0..500 {
        let error = optimizer.run_iteration().await.unwrap();
        if i % 50 == 0 {
            println!(
                "Iteration {}: error = {}, primitives = {}",
                i,
                error,
                optimizer.generate_irmf().split("val =").count() - 1
            );
        }
        // Allow for small variance due to stochastic sampling
        assert!(
            error <= last_error + 0.05,
            "Error increased too much at iteration {}: {} > {}",
            i,
            error,
            last_error
        );

        last_error = error;
    }

    println!("Final error: {}", last_error);
    assert!(last_error < 0.3); // Should have improved significantly

    let irmf = optimizer.generate_irmf();
    assert!(irmf.contains("mainModel4"));
}
