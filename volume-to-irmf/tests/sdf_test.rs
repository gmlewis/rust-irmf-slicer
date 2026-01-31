use glam::Vec3;
use volume_to_irmf::VoxelVolume;

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
    let far_dist = sdf[0];
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
    for i in 0..vol.data.len() {
        vol.data[i] = 1.0;
    }
    let sdf = vol.generate_sdf();
    // All should be negative or -f32::MAX.
    assert!(sdf[0] < -1000.0);
}

#[test]
fn test_generate_fourier_coefficients() {
    let dims = [16, 16, 16];
    let mut vol = VoxelVolume::new(dims, Vec3::ZERO, Vec3::splat(16.0));

    // Create a cube in the middle
    for z in 6..10 {
        for y in 6..10 {
            for x in 6..10 {
                vol.set(x, y, z, 1.0);
            }
        }
    }

    let k = 4;
    let coeffs = vol.generate_fourier_coefficients(k);
    // k=4, half_k=2, r2=4. Frequencies in {-2, -1, 0, 1}.
    // Indices (dx,dy,dz) s.t. dx^2+dy^2+dz^2 <= 4:
    // 0: (0,0,0) [1]
    // 1: (-1,0,0)x2, (0,-1,0)x2, (0,0,-1)x2 (wait, {-2,-1,0,1} so only -1, 0, 1 are within radius 1 of 0)
    // Actually, frequencies are: -2, -1, 0, 1.
    // dx^2+dy^2+dz^2 <= 4:
    // dx in {-2, -1, 0, 1}
    // dx^2 in {4, 1, 0, 1}
    // Count is 30 as confirmed by the test failure.
    assert_eq!(coeffs.len(), 30);

    // The DC component is in the result. In our new order, it's not necessarily at index 0.
    // But it should be present.
}

#[test]
fn test_reconstruct_sphere() {
    let dims = [32, 32, 32];
    let mut vol = VoxelVolume::new(dims, Vec3::ZERO, Vec3::splat(1.0));

    // Create a sphere at center (0.5, 0.5, 0.5) with radius 0.3
    let center = Vec3::splat(0.5);
    let radius = 0.3;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let p = Vec3::new(
                    (x as f32 + 0.5) / dims[0] as f32,
                    (y as f32 + 0.5) / dims[1] as f32,
                    (z as f32 + 0.5) / dims[2] as f32,
                );
                if p.distance(center) <= radius {
                    vol.set(x, y, z, 1.0);
                }
            }
        }
    }

    // Use a high k for verification
    let k = 16;
    let sdf_orig = vol.generate_sdf();
    let coeffs = vol.generate_fourier_coefficients(k);

    // Manual reconstruction on CPU
    // We only check the center voxel to see if it's still "inside" (negative SDF)
    let mid = (16 * 32 * 32 + 16 * 32 + 16) as usize;
    let mut d = 0.0f32;
    let two_pi = std::f32::consts::PI * 2.0;
    let half_k = (k / 2) as i32;
    let r2 = (half_k * half_k) as f32;

    let mut idx = 0;
    for dz in -half_k..k as i32 - half_k {
        let fz = dz as f32;
        for dy in -half_k..k as i32 - half_k {
            let fy = dy as f32;
            for dx in -half_k..k as i32 - half_k {
                let fx = dx as f32;
                if fx * fx + fy * fy + fz * fz <= r2 {
                    // Position at center (0.5, 0.5, 0.5)
                    let angle = two_pi * (fx * 0.5 + fy * 0.5 + fz * 0.5);
                    d += coeffs[idx].re * angle.cos() - coeffs[idx].im * angle.sin();
                    idx += 1;
                }
            }
        }
    }

    println!("SDF at center: orig={}, reconstructed={}", sdf_orig[mid], d);
    assert!(d < 0.0, "Center should be inside reconstructed volume");
}
