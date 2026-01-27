@group(0) @binding(0) var<storage, read> vertices: array<vec4f>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var voxel_volume: texture_storage_3d<r32float, write>;

struct Config {
    num_triangles: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    min_bound: vec4f,
    max_bound: vec4f,
};
@group(0) @binding(3) var<uniform> config: Config;

// Basic ray-triangle intersection (Möller-Trumbore)
fn intersect_ray_tri(orig: vec3f, dir: vec3f, v0: vec3f, v1: vec3f, v2: vec3f) -> f32 {
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(dir, edge2);
    let a = dot(edge1, h);
    if (a > -0.000001 && a < 0.000001) { return -1.0; }
    let f = 1.0 / a;
    let s = orig - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return -1.0; }
    let q = cross(s, edge1);
    let v = f * dot(dir, q);
    if (v < 0.0 || u + v > 1.0) { return -1.0; }
    let t = f * dot(edge2, q);
    return t;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(voxel_volume);
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }

    let x = global_id.x;
    let y = global_id.y;

    // We'll collect intersection parity in a bitset (up to 1024 Z-resolution)
    var bits: array<u32, 32>;
    for (var i = 0u; i < 32u; i++) {
        bits[i] = 0u;
    }

    let world_size = config.max_bound.xyz - config.min_bound.xyz;
    let voxel_size = world_size / vec3f(dims);
    
    // Ray origin below the bottom of the (X, Y) column.
    // Use a small offset in X and Y to avoid hitting edges exactly.
    let orig = vec3f(
        config.min_bound.x + (f32(x) + 0.50013) * voxel_size.x,
        config.min_bound.y + (f32(y) + 0.50013) * voxel_size.y,
        config.min_bound.z - 0.1 * world_size.z // Start 10% below the min bound
    );
    let ray_dir = vec3f(0.0, 0.0, 1.0);
    
    for (var i = 0u; i < config.num_triangles; i++) {
        let v0 = vertices[indices[i * 3u + 0u]].xyz;
        let v1 = vertices[indices[i * 3u + 1u]].xyz;
        let v2 = vertices[indices[i * 3u + 2u]].xyz;
        
        let t = intersect_ray_tri(orig, ray_dir, v0, v1, v2);
        if (t >= 0.0) {
            let world_z = orig.z + t;
            let z_world_rel = (world_z - config.min_bound.z) / voxel_size.z;
            let z_idx = i32(round(z_world_rel));
            if (z_idx >= 0 && u32(z_idx) < dims.z) {
                bits[u32(z_idx) / 32u] ^= (1u << (u32(z_idx) % 32u));
            }
        }
    }

    // Scan through bits and fill voxels along the Z column
    var inside = false;
    for (var z = 0u; z < dims.z; z++) {
        // If an intersection happened in this voxel (or at its boundary), flip inside/outside
        if (((bits[z / 32u] >> (z % 32u)) & 1u) == 1u) {
            inside = !inside;
        }
        textureStore(voxel_volume, vec3u(x, y, z), vec4f(f32(inside), 0.0, 0.0, 0.0));
    }
}