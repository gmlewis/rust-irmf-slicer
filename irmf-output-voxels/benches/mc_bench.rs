use criterion::{criterion_group, criterion_main, Criterion};
use irmf_output_voxels::BinVox;

fn bench_marching_cubes(c: &mut Criterion) {
    let nx = 100;
    let ny = 100;
    let nz = 100;
    let mut b = BinVox::new(nx, ny, nz, [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    
    // Fill with a sphere
    for z in 0..nz {
        let fz = (z as f32 / nz as f32) * 10.0 - 5.0;
        for y in 0..ny {
            let fy = (y as f32 / ny as f32) * 10.0 - 5.0;
            for x in 0..nx {
                let fx = (x as f32 / nx as f32) * 10.0 - 5.0;
                if fx*fx + fy*fy + fz*fz <= 4.0*4.0 {
                    b.set(x, y, z);
                }
            }
        }
    }

    c.bench_function("marching_cubes_100x100x100", |b_iter| {
        b_iter.iter(|| {
            b.marching_cubes::<fn(usize, usize)>(None);
        })
    });
}

criterion_group!(benches, bench_marching_cubes);
criterion_main!(benches);
