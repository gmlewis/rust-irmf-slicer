struct Triangle {
    normal: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    v3: vec4<f32>,
}

struct Output {
    count: atomic<u32>,
    triangles: array<Triangle>,
}

struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    min_x: f32,
    min_y: f32,
    min_z: f32,
    dx: f32,
    dy: f32,
    dz: f32,
}

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var<storage, read> tri_table: array<i32>;
@group(0) @binding(2) var<storage, read_write> output: Output;
@group(0) @binding(3) var<uniform> params: Params;

fn get_voxel(x: i32, y: i32, z: i32) -> bool {
    if (x < 0 || y < 0 || z < 0 || u32(x) >= params.nx || u32(y) >= params.ny || u32(z) >= params.nz) {
        return false;
    }
    let idx = u32(z) * params.nx * params.ny + u32(y) * params.nx + u32(x);
    let word = voxels[idx / 32u];
    let bit = idx % 32u;
    return (word & (1u << bit)) != 0u;
}

fn get_pos(x: i32, y: i32, z: i32, v_idx: i32) -> vec3<f32> {
    var p: vec3<f32>;
    switch v_idx {
        case 0: { p = vec3<f32>(0.0, 0.0, 0.0); }
        case 1: { p = vec3<f32>(1.0, 0.0, 0.0); }
        case 2: { p = vec3<f32>(1.0, 1.0, 0.0); }
        case 3: { p = vec3<f32>(0.0, 1.0, 0.0); }
        case 4: { p = vec3<f32>(0.0, 0.0, 1.0); }
        case 5: { p = vec3<f32>(1.0, 0.0, 1.0); }
        case 6: { p = vec3<f32>(1.0, 1.0, 1.0); }
        case 7: { p = vec3<f32>(0.0, 1.0, 1.0); }
        default: { p = vec3<f32>(0.0, 0.0, 0.0); }
    }
    return vec3<f32>(
        params.min_x + (f32(x) + p.x + 0.5) * params.dx,
        params.min_y + (f32(y) + p.y + 0.5) * params.dy,
        params.min_z + (f32(z) + p.z + 0.5) * params.dz
    );
}

fn interp_edge(x: i32, y: i32, z: i32, edge: i32) -> vec3<f32> {
    var v1_idx: i32;
    var v2_idx: i32;

    switch edge {
        case 0: { v1_idx = 0; v2_idx = 1; }
        case 1: { v1_idx = 1; v2_idx = 2; }
        case 2: { v1_idx = 2; v2_idx = 3; }
        case 3: { v1_idx = 3; v2_idx = 0; }
        case 4: { v1_idx = 4; v2_idx = 5; }
        case 5: { v1_idx = 5; v2_idx = 6; }
        case 6: { v1_idx = 6; v2_idx = 7; }
        case 7: { v1_idx = 7; v2_idx = 4; }
        case 8: { v1_idx = 0; v2_idx = 4; }
        case 9: { v1_idx = 1; v2_idx = 5; }
        case 10: { v1_idx = 2; v2_idx = 6; }
        case 11: { v1_idx = 3; v2_idx = 7; }
        default: { v1_idx = 0; v2_idx = 0; }
    }
    
    let p1 = get_pos(x, y, z, v1_idx);
    let p2 = get_pos(x, y, z, v2_idx);
    return (p1 + p2) * 0.5;
}

@compute @workgroup_size(8, 8, 2)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = i32(global_id.x) - 1;
    let y = i32(global_id.y) - 1;
    let z = i32(global_id.z) - 1;

    if (x >= i32(params.nx) || y >= i32(params.ny) || z >= i32(params.nz)) {
        return;
    }

    var cube_index = 0u;
    if (get_voxel(x, y, z)) { cube_index |= 1u; }
    if (get_voxel(x + 1, y, z)) { cube_index |= 2u; }
    if (get_voxel(x + 1, y + 1, z)) { cube_index |= 4u; }
    if (get_voxel(x, y + 1, z)) { cube_index |= 8u; }
    if (get_voxel(x, y, z + 1)) { cube_index |= 16u; }
    if (get_voxel(x + 1, y, z + 1)) { cube_index |= 32u; }
    if (get_voxel(x + 1, y + 1, z + 1)) { cube_index |= 64u; }
    if (get_voxel(x, y + 1, z + 1)) { cube_index |= 128u; }

    if (cube_index == 0u || cube_index == 255u) {
        return;
    }

    for (var i = 0u; i < 16u; i += 3u) {
        let e1 = tri_table[cube_index * 16u + i];
        if (e1 == -1) { break; }
        let e2 = tri_table[cube_index * 16u + i + 1u];
        let e3 = tri_table[cube_index * 16u + i + 2u];

        let v1 = interp_edge(x, y, z, e1);
        let v2 = interp_edge(x, y, z, e2);
        let v3 = interp_edge(x, y, z, e3);

        let normal = normalize(cross(v2 - v1, v3 - v1));
        
        let idx = atomicAdd(&output.count, 1u);
        // Check for overflow? (count is compared with max_triangles in host code)
        if (idx < arrayLength(&output.triangles)) {
            output.triangles[idx].normal = vec4<f32>(normal, 0.0);
            output.triangles[idx].v1 = vec4<f32>(v1, 1.0);
            output.triangles[idx].v2 = vec4<f32>(v2, 1.0);
            output.triangles[idx].v3 = vec4<f32>(v3, 1.0);
        }
    }
}
