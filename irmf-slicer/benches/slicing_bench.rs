use criterion::{Criterion, criterion_group, criterion_main};
use irmf_slicer::{IrmfModel, Slicer, WgpuRenderer};
use pollster::block_on;

fn bench_slicing(c: &mut Criterion) {
    let renderer = block_on(WgpuRenderer::new()).expect("Failed to create WgpuRenderer");

    // 1. Benchmark simple sphere
    let sphere_data = b"/*{
  \"irmf\": \"1.0\",
  \"materials\": [\"PLA\"],
  \"max\": [5,5,5],
  \"min\": [-5,-5,-5],
  \"units\": \"mm\"
}*/
void mainModel4(out vec4 materials, in vec3 xyz) {
  materials[0] = length(xyz) <= 5.0 ? 1.0 : 0.0;
}";
    let sphere_model = IrmfModel::new(sphere_data).unwrap();
    let mut sphere_slicer = Slicer::new(sphere_model, renderer, 1000.0, 1000.0, 1000.0);
    sphere_slicer.prepare_render_z().unwrap();

    c.bench_function("slice_sphere_10x10x10", |b| {
        b.iter(|| {
            sphere_slicer
                .render_z_slices(1, |_, _, _, _| Ok(()))
                .unwrap();
        })
    });

    // 2. Benchmark complex soapdish
    let soapdish_data = include_bytes!("../../examples/015-soapdish/soapdish-step-10.irmf");
    let soapdish_model = IrmfModel::new(soapdish_data).unwrap();

    // Re-use the renderer (we need to re-init and prepare though)
    let renderer = block_on(WgpuRenderer::new()).expect("Failed to create WgpuRenderer");
    let mut soapdish_slicer = Slicer::new(soapdish_model, renderer, 1000.0, 1000.0, 1000.0);
    soapdish_slicer.prepare_render_z().unwrap();

    c.bench_function("slice_soapdish_115x115x30", |b| {
        b.iter(|| {
            soapdish_slicer
                .render_z_slices(1, |_, _, _, _| Ok(()))
                .unwrap();
        })
    });
}

criterion_group!(benches, bench_slicing);
criterion_main!(benches);
