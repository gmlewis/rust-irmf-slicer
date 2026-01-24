@group(0) @binding(0) var target_volume: texture_3d<f32>;
@group(0) @binding(1) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(2) var sdf_out: texture_storage_3d<r32float, write>;

struct Primitive {
    pos: vec3f,
    prim_type: u32,
    size: vec3f,
    op: u32,
}

struct Config {
    num_primitives: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

fn sd_sphere(p: vec3f, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sd_box(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn evaluate_model(p: vec3f) -> f32 {
    var val = 0.0;
    for (var i = 0u; i < config.num_primitives; i++) {
        let prim = primitives[i];
        let p_local = p - prim.pos;
        var dist = 0.0;
        if (prim.prim_type == 0u) {
            dist = sd_sphere(p_local, prim.size.x);
        } else {
            dist = sd_box(p_local, prim.size);
        }
        let occupancy = select(0.0, 1.0, dist <= 0.0);
        if (prim.op == 0u) {
            val = max(val, occupancy);
        } else {
            val = min(val, 1.0 - occupancy);
        }
    }
    return val;
}

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(target_volume);
    if (any(global_id >= dims)) { return; }

    let p = (vec3f(global_id) + 0.5) / vec3f(dims);
    let target_val = textureLoad(target_volume, global_id, 0).r;
    let current_val = evaluate_model(p);

    // We want to fill areas where target is 1 but current is 0
    let diff = target_val - current_val;
    
    if (diff < 0.5) {
        textureStore(sdf_out, global_id, vec4f(-1.0, 0.0, 0.0, 0.0));
        return;
    }

    // Still O(N^6) but only for relevant voxels. 
    // For a 32x32x32 test, it might be acceptable.
    var min_dist_sq = 100.0;
    for (var z = 0u; z < dims.z; z++) {
        for (var y = 0u; y < dims.y; y++) {
            for (var x = 0u; x < dims.x; x++) {
                let other_target = textureLoad(target_volume, vec3u(x, y, z), 0).r;
                let other_current = evaluate_model((vec3f(x, y, z) + 0.5) / vec3f(dims));
                if (other_target - other_current < 0.5) {
                    let other_p = (vec3f(x, y, z) + 0.5) / vec3f(dims);
                    let delta = p - other_p;
                    min_dist_sq = min(min_dist_sq, dot(delta, delta));
                }
            }
        }
    }

    textureStore(sdf_out, global_id, vec4f(sqrt(min_dist_sq), 0.0, 0.0, 0.0));
}