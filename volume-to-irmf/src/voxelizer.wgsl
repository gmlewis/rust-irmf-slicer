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
    if (a > -0.00001 && a < 0.00001) { return -1.0; }
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

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(voxel_volume);
    if (any(global_id >= dims)) { return; }

    let p = config.min_bound.xyz + (vec3f(global_id) + 0.5) / vec3f(dims) * (config.max_bound.xyz - config.min_bound.xyz);
    
    // Ray cast along +Z
    let ray_dir = vec3f(0.0, 0.0, 1.0);
    var intersections = 0u;
    
    for (var i = 0u; i < config.num_triangles; i++) {
        let v0 = vertices[indices[i * 3u + 0u]].xyz;
        let v1 = vertices[indices[i * 3u + 1u]].xyz;
        let v2 = vertices[indices[i * 3u + 2u]].xyz;
        
        let t = intersect_ray_tri(p, ray_dir, v0, v1, v2);
        if (t > 0.0) {
            intersections++;
        }
    }

    let val = f32(intersections % 2u);
    textureStore(voxel_volume, global_id, vec4f(val, 0.0, 0.0, 0.0));
}
