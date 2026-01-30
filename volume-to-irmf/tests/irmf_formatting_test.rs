use glam::Vec3;
use volume_to_irmf::{Optimizer, VoxelVolume};

async fn create_test_optimizer() -> Optimizer {
    let dims = [4, 4, 4];
    let mut volume = VoxelVolume::new(dims, Vec3::ZERO, Vec3::splat(1.0));
    // Create a tiny 2x2x2 cube in the middle
    for z in 1..3 {
        for y in 1..3 {
            for x in 1..3 {
                volume.set(x, y, z, 1.0);
            }
        }
    }
    Optimizer::new(volume, false).await.unwrap()
}

#[tokio::test]
async fn test_formatting_lossless_glsl() {
    let mut opt = create_test_optimizer().await;
    opt.run_lossless().await.unwrap();
    let irmf = opt.generate_irmf("glsl".to_string());

    let expected = r#"/*{
  "irmf": "1.0",
  "language": "glsl",
  "materials": ["Material"],
  "min": [0.0000, 0.0000, 0.0000],
  "max": [1.0000, 1.0000, 1.0000],
  "notes": "Final Lossless Cuboids",
  "units": "mm"
}*/

const vec3 MIN_BOUND = vec3(0.0000, 0.0000, 0.0000);
const vec3 MAX_BOUND = vec3(1.0000, 1.0000, 1.0000);
const vec3 DIMS = vec3(4.0, 4.0, 4.0);
const vec3 VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const vec4 solidMaterial = vec4(1.0, 0.0, 0.0, 0.0);

bool cuboid(ivec3 v, ivec3 b_min, ivec3 b_max) {
    return all(greaterThanEqual(v, b_min)) && all(lessThanEqual(v, b_max));
}

vec4 bz1by1Case(ivec3 vi) {
    if (cuboid(vi, ivec3(1, 1, 1), ivec3(2, 2, 2))) { return solidMaterial; }
    return vec4(0,0,0,0);
}

vec4 bz1by2Case(ivec3 vi) {
    if (cuboid(vi, ivec3(1, 1, 1), ivec3(2, 2, 2))) { return solidMaterial; }
    return vec4(0,0,0,0);
}

vec4 bz2by1Case(ivec3 vi) {
    if (cuboid(vi, ivec3(1, 1, 1), ivec3(2, 2, 2))) { return solidMaterial; }
    return vec4(0,0,0,0);
}

vec4 bz2by2Case(ivec3 vi) {
    if (cuboid(vi, ivec3(1, 1, 1), ivec3(2, 2, 2))) { return solidMaterial; }
    return vec4(0,0,0,0);
}


vec4 bz1Case(ivec3 vi, int by) {
    if (by == 1) { return bz1by1Case(vi); }
    if (by == 2) { return bz1by2Case(vi); }
    return vec4(0,0,0,0);
}

vec4 bz2Case(ivec3 vi, int by) {
    if (by == 1) { return bz2by1Case(vi); }
    if (by == 2) { return bz2by2Case(vi); }
    return vec4(0,0,0,0);
}

void mainModel4(out vec4 materials, in vec3 xyz) {
    vec3 v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    ivec3 vi = ivec3(floor(v));
    int bz = vi.z / 1;
    int by = vi.y / 1;
    if (bz == 1) { materials = bz1Case(vi, by); return; }
    if (bz == 2) { materials = bz2Case(vi, by); return; }

    materials = vec4(0,0,0,0);
}
"#;

    assert_eq!(irmf, expected, "Lossless GLSL formatting mismatch");
}

#[tokio::test]
async fn test_formatting_lossless_wgsl() {
    let mut opt = create_test_optimizer().await;
    opt.run_lossless().await.unwrap();
    let irmf = opt.generate_irmf("wgsl".to_string());

    let expected = r#"/*{
  "irmf": "1.0",
  "language": "wgsl",
  "materials": ["Material"],
  "min": [0.0000, 0.0000, 0.0000],
  "max": [1.0000, 1.0000, 1.0000],
  "notes": "Final Lossless Cuboids",
  "units": "mm"
}*/

const MIN_BOUND = vec3f(0.0000, 0.0000, 0.0000);
const MAX_BOUND = vec3f(1.0000, 1.0000, 1.0000);
const DIMS = vec3f(4.0, 4.0, 4.0);
const VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const solidMaterial = vec4f(1.0, 0.0, 0.0, 0.0);

fn cuboid(v: vec3i, b_min: vec3i, b_max: vec3i) -> bool {
    return all(v >= b_min) && all(v <= b_max);
}

fn bz1by1Case(vi: vec3i) -> vec4f {
    if (cuboid(vi, vec3i(1, 1, 1), vec3i(2, 2, 2))) { return solidMaterial; }
    return vec4f(0,0,0,0);
}

fn bz1by2Case(vi: vec3i) -> vec4f {
    if (cuboid(vi, vec3i(1, 1, 1), vec3i(2, 2, 2))) { return solidMaterial; }
    return vec4f(0,0,0,0);
}

fn bz2by1Case(vi: vec3i) -> vec4f {
    if (cuboid(vi, vec3i(1, 1, 1), vec3i(2, 2, 2))) { return solidMaterial; }
    return vec4f(0,0,0,0);
}

fn bz2by2Case(vi: vec3i) -> vec4f {
    if (cuboid(vi, vec3i(1, 1, 1), vec3i(2, 2, 2))) { return solidMaterial; }
    return vec4f(0,0,0,0);
}

fn bz1Case(vi: vec3i, by: i32) -> vec4f {
    if (by == 1) { return bz1by1Case(vi); }
    if (by == 2) { return bz1by2Case(vi); }
    return vec4f(0,0,0,0);
}
fn bz2Case(vi: vec3i, by: i32) -> vec4f {
    if (by == 1) { return bz2by1Case(vi); }
    if (by == 2) { return bz2by2Case(vi); }
    return vec4f(0,0,0,0);
}

fn mainModel4(xyz: vec3f) -> vec4f {
    let v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    let vi = vec3i(floor(v + vec3f(0.5)));
    let iy = i32(floor(v.y));
    let iz = i32(floor(v.z));
    let bz = vi.z / 1;
    let by = vi.y / 1;
    if (bz == 1) { return bz1Case(vi, by); }
    if (bz == 2) { return bz2Case(vi, by); }

    return vec4f(0.0, 0.0, 0.0, 0.0);
}
"#;

    assert_eq!(irmf, expected, "Lossless WGSL formatting mismatch");
}

#[tokio::test]
async fn test_formatting_fourier_glsl() {
    let mut opt = create_test_optimizer().await;
    // k=2 results in 8 coefficients
    opt.run_fourier(2).await.unwrap();
    let irmf = opt.generate_fourier_irmf("glsl".to_string());

    let expected = r#"/*{
  "irmf": "1.0",
  "language": "glsl",
  "materials": ["Material"],
  "min": [0.0000, 0.0000, 0.0000],
  "max": [1.0000, 1.0000, 1.0000],
  "notes": "Fourier Approximation",
  "units": "mm"
}*/

const vec3 MIN_BOUND = vec3(0.0000, 0.0000, 0.0000);
const vec3 MAX_BOUND = vec3(1.0000, 1.0000, 1.0000);
const vec3 DIMS = vec3(4.0, 4.0, 4.0);
const vec3 VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const vec4 solidMaterial = vec4(1.0, 0.0, 0.0, 0.0);

bool cuboid(ivec3 v, ivec3 b_min, ivec3 b_max) {
    return all(greaterThanEqual(v, b_min)) && all(lessThanEqual(v, b_max));
}

const float coeffs_re[8] = float[](
    -0.046544, 0.000000, 0.000000, 0.196642, 0.000000, 0.196642, 0.196642, 0.996836
);

const float coeffs_im[8] = float[](
    -0.046544, 0.105135, 0.105135, -0.196642, 0.105135, -0.196642, -0.196642, 0.000000
);

void mainModel4(out vec4 materials, in vec3 xyz) {
    vec3 v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    ivec3 vi = ivec3(floor(v));

    float d = 0.0;
    float TWO_PI = 6.28318530718;
    int half_k = 1;

    // Precompute 1D basis functions
    float cos_x[2], sin_x[2];
    float cos_y[2], sin_y[2];
    float cos_z[2], sin_z[2];

    for (int i = 0; i < 2; i++) {
        float f = float(i - half_k);
        float ax = TWO_PI * f * v.x / DIMS.x;
        float ay = TWO_PI * f * v.y / DIMS.y;
        float az = TWO_PI * f * v.z / DIMS.z;
        cos_x[i] = cos(ax); sin_x[i] = sin(ax);
        cos_y[i] = cos(ay); sin_y[i] = sin(ay);
        cos_z[i] = cos(az); sin_z[i] = sin(az);
    }

    for (int dz = 0; dz < 2; dz++) {
        for (int dy = 0; dy < 2; dy++) {
            float cycz = cos_y[dy] * cos_z[dz];
            float cysz = cos_y[dy] * sin_z[dz];
            float sycz = sin_y[dy] * cos_z[dz];
            float sysz = sin_y[dy] * sin_z[dz];

            float cos_bc = cycz - sysz;
            float sin_bc = sycz + cysz;

            for (int dx = 0; dx < 2; dx++) {
                int idx = dz * 2 * 2 + dy * 2 + dx;

                float cos_abc = cos_x[dx] * cos_bc - sin_x[dx] * sin_bc;
                float sin_abc = sin_x[dx] * cos_bc + cos_x[dx] * sin_bc;

                d += coeffs_re[idx] * cos_abc - coeffs_im[idx] * sin_abc;
            }
        }
    }
    if (d < 0.0) { materials = solidMaterial; return; }

    materials = vec4(0,0,0,0);
}
"#;

    assert_eq!(irmf, expected, "Fourier GLSL formatting mismatch");
}

#[tokio::test]
async fn test_formatting_fourier_wgsl() {
    let mut opt = create_test_optimizer().await;
    opt.run_fourier(2).await.unwrap();
    let irmf = opt.generate_fourier_irmf("wgsl".to_string());

    let expected = r#"/*{
  "irmf": "1.0",
  "language": "wgsl",
  "materials": ["Material"],
  "min": [0.0000, 0.0000, 0.0000],
  "max": [1.0000, 1.0000, 1.0000],
  "notes": "Fourier Approximation",
  "units": "mm"
}*/

const MIN_BOUND = vec3f(0.0000, 0.0000, 0.0000);
const MAX_BOUND = vec3f(1.0000, 1.0000, 1.0000);
const DIMS = vec3f(4.0, 4.0, 4.0);
const VOXEL_SIZE = (MAX_BOUND - MIN_BOUND) / DIMS;
const solidMaterial = vec4f(1.0, 0.0, 0.0, 0.0);

fn cuboid(v: vec3i, b_min: vec3i, b_max: vec3i) -> bool {
    return all(v >= b_min) && all(v <= b_max);
}

const coeffs_re = array<f32, 8>(
    -0.046544, 0.000000, 0.000000, 0.196642, 0.000000, 0.196642, 0.196642, 0.996836
);

const coeffs_im = array<f32, 8>(
    -0.046544, 0.105135, 0.105135, -0.196642, 0.105135, -0.196642, -0.196642, 0.000000
);

fn mainModel4(xyz: vec3f) -> vec4f {
    let v = (xyz - MIN_BOUND) / VOXEL_SIZE;
    let vi = vec3i(floor(v + vec3f(0.5)));
    let iy = i32(floor(v.y));
    let iz = i32(floor(v.z));

    var d: f32 = 0.0;
    const TWO_PI: f32 = 6.28318530718;
    const half_k: i32 = 1;

    var cos_x: array<f32, 2>; var sin_x: array<f32, 2>;
    var cos_y: array<f32, 2>; var sin_y: array<f32, 2>;
    var cos_z: array<f32, 2>; var sin_z: array<f32, 2>;

    for (var i: i32 = 0; i < 2; i++) {
        let f = f32(i - half_k);
        let ax = TWO_PI * f * v.x / DIMS.x;
        let ay = TWO_PI * f * v.y / DIMS.y;
        let az = TWO_PI * f * v.z / DIMS.z;
        cos_x[i] = cos(ax); sin_x[i] = sin(ax);
        cos_y[i] = cos(ay); sin_y[i] = sin(ay);
        cos_z[i] = cos(az); sin_z[i] = sin(az);
    }

    for (var dz: i32 = 0; dz < 2; dz++) {
        for (var dy: i32 = 0; dy < 2; dy++) {
            let cycz = cos_y[dy] * cos_z[dz];
            let cysz = cos_y[dy] * sin_z[dz];
            let sycz = sin_y[dy] * cos_z[dz];
            let sysz = sin_y[dy] * sin_z[dz];

            for (var dx: i32 = 0; dx < 2; dx++) {
                let idx = dz * 2 * 2 + dy * 2 + dx;

                // Use trigonometric identities to avoid cos/sin inside the inner loop
                // cos(a+b+c) = cos(a)cos(b+c) - sin(a)sin(b+c)
                // sin(a+b+c) = sin(a)cos(b+c) + cos(a)sin(b+c)
                // where:
                // cos(b+c) = cos(b)cos(c) - sin(b)sin(c)
                // sin(b+c) = sin(b)cos(c) + cos(b)sin(c)

                let cos_bc = cycz - sysz;
                let sin_bc = sycz + cysz;

                let cos_abc = cos_x[dx] * cos_bc - sin_x[dx] * sin_bc;
                let sin_abc = sin_x[dx] * cos_bc + cos_x[dx] * sin_bc;

                d += coeffs_re[idx] * cos_abc - coeffs_im[idx] * sin_abc;
            }
        }
    }
    if (d < 0.0) { return solidMaterial; }

    return vec4f(0.0, 0.0, 0.0, 0.0);
}
"#;

    assert_eq!(irmf, expected, "Fourier WGSL formatting mismatch");
}
