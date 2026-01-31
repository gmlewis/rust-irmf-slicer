@group(0) @binding(0) var<storage, read> volume: array<f32>;
@group(0) @binding(1) var<storage, read_write> projections: array<f32>;
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
    let r_idx = id.x;
    let a_idx = id.y;
    let z = id.z;

    if (r_idx >= params.nr || a_idx >= params.n_angles || z >= params.nz) {
        return;
    }

    let angle_rad = params.angles[a_idx].x;
    let cos_a = cos(angle_rad);
    let sin_a = sin(angle_rad);

    let center_x = (f32(params.nx) - 1.0) / 2.0;
    let center_y = (f32(params.ny) - 1.0) / 2.0;
    let center_r = (f32(params.nr) - 1.0) / 2.0;

    let r = f32(r_idx) - center_r;
    var sum = 0.0;

    for (var t_idx: u32 = 0; t_idx < params.nr; t_idx = t_idx + 1) {
        let t = f32(t_idx) - center_r;
        let x = r * cos_a - t * sin_a + center_x;
        let y = r * sin_a + t * cos_a + center_y;

        if (x >= 0.0 && x < f32(params.nx - 1) && y >= 0.0 && y < f32(params.ny - 1)) {
            let x0 = u32(floor(x));
            let y0 = u32(floor(y));
            let x1 = x0 + 1;
            let y1 = y0 + 1;
            let dx = x - f32(x0);
            let dy = y - f32(y0);

            // Correct indexing: (x * ny + y) * nz + z
            let v00 = volume[(x0 * params.ny + y0) * params.nz + z];
            let v10 = volume[(x1 * params.ny + y0) * params.nz + z];
            let v01 = volume[(x0 * params.ny + y1) * params.nz + z];
            let v11 = volume[(x1 * params.ny + y1) * params.nz + z];

            sum = sum + (1.0 - dx) * (1.0 - dy) * v00
                      + dx * (1.0 - dy) * v10
                      + (1.0 - dx) * dy * v01
                      + dx * dy * v11;
        }
    }

    // Correct indexing: (r * n_angles + a) * nz + z
    projections[(r_idx * params.n_angles + a_idx) * params.nz + z] = sum;
}