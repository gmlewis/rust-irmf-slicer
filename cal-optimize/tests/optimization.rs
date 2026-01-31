use cal_optimize::{CalOptimizer, TargetVolume};
use irmf_slicer::Slicer;
use irmf_slicer::irmf::IrmfModel;
use irmf_slicer::mock_renderer::MockRenderer;

#[test]
fn test_optimization_loop() {
    let data =
        br#"/*{"irmf":"1.0","materials":["PLA"],"max":[2,2,2],"min":[-2,-2,-2],"units":"mm"}*/
    void mainModel4(out vec4 materials, in vec3 xyz) { 
        if (length(xyz) < 1.0) {
            materials[0] = 1.0; 
        } else {
            materials[0] = 0.0;
        }
    }"#;

    let model = IrmfModel::new(data).expect("Failed to parse IRMF model");
    let renderer = MockRenderer::new();
    let mut slicer = Slicer::new(model, renderer, 1000.0, 1000.0, 1000.0);

    let target = TargetVolume::from_irmf(&mut slicer, 0).expect("Failed to create target volume");
    // With 1000 microns (1mm) resolution and range [-2, 2], we expect 4 voxels per dim.
    assert_eq!(target.data.dim(), (4, 4, 4));

    let angles = vec![0.0, 45.0, 90.0, 135.0];
    let mut optimizer = CalOptimizer::new(target, angles, cal_optimize::CpuProjector);
    optimizer.max_iter = 2; // Keep it fast for testing
    let (opt_b, errors) = optimizer.run();

    // opt_b dim: (nr, n_angles, nz)
    // nr is calculated as ceil(sqrt(4^2 + 4^2)) = ceil(sqrt(32)) = 6
    assert_eq!(opt_b.dim().1, 4); // 4 angles
    assert_eq!(errors.len(), 2);

    println!("VER Errors: {:?}", errors);
}
