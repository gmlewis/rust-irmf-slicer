@group(0) @binding(0) var<storage, read> projections: array<f32>;
@group(0) @binding(1) var<storage, read_write> volume: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    nr: u32,
    n_angles: u32,
    _p1: u32,
    _p2: u32,
    _p3: u32,
    angles: array<vec4<f32>, 128>,
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x_idx = id.x;
    let y_idx = id.y;
    let z = id.z;

    if (x_idx >= params.nx || y_idx >= params.ny || z >= params.nz) {
        return;
    }

    let center_x = (f32(params.nx) - 1.0) / 2.0;
    let center_y = (f32(params.ny) - 1.0) / 2.0;
    let center_r = (f32(params.nr) - 1.0) / 2.0;

    let x = f32(x_idx) - center_x;
    let y = f32(y_idx) - center_y;

    var sum = 0.0;
    for (var a_idx: u32 = 0; a_idx < params.n_angles; a_idx = a_idx + 1) {
        let angle_rad = params.angles[a_idx].x;
        let cos_a = cos(angle_rad);
        let sin_a = sin(angle_rad);

        let r = x * cos_a + y * sin_a + center_r;

        if (r >= 0.0 && r < f32(params.nr - 1)) {
            let r0 = u32(floor(r));
            let r1 = r0 + 1;
            let dr = r - f32(r0);

            // Correct indexing: (r * n_angles + a) * nz + z
            let v0 = projections[(r0 * params.n_angles + a_idx) * params.nz + z];
            let v1 = projections[(r1 * params.n_angles + a_idx) * params.nz + z];
            sum = sum + (1.0 - dr) * v0 + dr * v1;
        }
    }

    // Correct indexing: (x * ny + y) * nz + z
    volume[(x_idx * params.ny + y_idx) * params.nz + z] = sum / f32(params.n_angles);
}