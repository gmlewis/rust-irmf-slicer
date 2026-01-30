use volume_to_irmf::VoxelVolume;
use glam::Vec3;

#[test]
fn test_generate_sdf_cube() {
    let dims = [10, 10, 10];
    let mut vol = VoxelVolume::new(dims, Vec3::ZERO, Vec3::splat(10.0));
    
    // Create a 2x2x2 cube in the middle: indices 4, 5
    for z in 4..6 {
        for y in 4..6 {
            for x in 4..6 {
                vol.set(x, y, z, 1.0);
            }
        }
    }
    
    let sdf = vol.generate_sdf();
    
    // Check points inside (indices 4, 5)
    // They are 1 voxel away from the outside (indices 3, 6)
    assert_eq!(sdf[(4 * 100 + 4 * 10 + 4) as usize], -1.0);
    assert_eq!(sdf[(5 * 100 + 5 * 10 + 5) as usize], -1.0);
    
    // Check points just outside
    // (3,4,4) is outside, but adjacent to (4,4,4)
    assert_eq!(sdf[(3 * 100 + 4 * 10 + 4) as usize], 1.0);
    assert_eq!(sdf[(6 * 100 + 4 * 10 + 4) as usize], 1.0);

    // Check a far away point
    // (0,0,0) is far from (4,4,4)
    // dx=4, dy=4, dz=4 -> dist = sqrt(16+16+16) = sqrt(48) approx 6.928
    let far_dist = sdf[(0 * 100 + 0 * 10 + 0) as usize];
    assert!((far_dist - 48.0f32.sqrt()).abs() < 1e-5);
}

#[test]
fn test_generate_sdf_empty() {
    let dims = [5, 5, 5];
    let vol = VoxelVolume::new(dims, Vec3::ZERO, Vec3::ONE);
    let sdf = vol.generate_sdf();
    // All should be positive or f32::MAX if we don't have inside.
    // In my implementation, if no inside points, dist_to_inside stays f32::MAX.
    // So SDF = sqrt(f32::MAX) = something very large.
    assert!(sdf[0] > 1000.0);
}

#[test]
fn test_generate_sdf_full() {
    let dims = [5, 5, 5];
    let mut vol = VoxelVolume::new(dims, Vec3::ZERO, Vec3::ONE);
    for i in 0..vol.data.len() { vol.data[i] = 1.0; }
    let sdf = vol.generate_sdf();
    // All should be negative or -f32::MAX.
    assert!(sdf[0] < -1000.0);
}
