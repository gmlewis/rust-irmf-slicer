struct Primitive {
    pos: vec3f,
    prim_type: u32,
    size: vec3f,
    op: u32,
}

struct ErrorResult {
    mse_sum: f32,
    iou_min_sum: f32,
    iou_max_sum: f32,
    padding: f32,
}

@group(0) @binding(0) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(1) var target_volume: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<ErrorResult>;

struct Config {
    num_samples: u32,
    num_primitives: u32,
    seed: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

// Simple random number generator
fn hash(u: u32) -> u32 {
    var x = u;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

fn rand_vec3(index: u32) -> vec3f {
    let h1 = hash(index ^ config.seed);
    let h2 = hash(h1);
    let h3 = hash(h2);
    return vec3f(
        f32(h1) / 4294967295.0,
        f32(h2) / 4294967295.0,
        f32(h3) / 4294967295.0
    );
}

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
        if (prim.prim_type == 0u) { // Sphere
            dist = sd_sphere(p_local, prim.size.x);
        } else { // Cube
            dist = sd_box(p_local, prim.size);
        }
        
        // Convert distance to occupancy (0 or 1)
        // For smooth gradients in optimization, we might use a sigmoid or similar
        // but for now, hard step.
        let occupancy = select(0.0, 1.0, dist <= 0.0);
        
        if (prim.op == 0u) { // Union
            val = max(val, occupancy);
        } else { // Difference
            val = min(val, 1.0 - occupancy);
        }
    }
    return val;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= config.num_samples) {
        return;
    }

    // Sample point in [0, 1] normalized coordinates
    let uvw = rand_vec3(index);
    
    // Sample target volume
    // textureSample is not available in compute shaders for storage textures,
    // but target_volume is a texture_3d<f32>.
    // Wait, we need to use textureLoad with integer coordinates or use a sampler.
    let dims = textureDimensions(target_volume);
    let coords = vec3<u32>(uvw * vec3f(dims));
    let target_val = textureLoad(target_volume, coords, 0).r;

    // Evaluate candidate model
    // map uvw to world coordinates if needed, but for now let's assume primitives are in [0, 1]
    let candidate_val = evaluate_model(uvw);

    // Calculate error components
    let diff = target_val - candidate_val;
    let mse = diff * diff;
    let iou_min = min(target_val, candidate_val);
    let iou_max = max(target_val, candidate_val);

    results[index].mse_sum = mse;
    results[index].iou_min_sum = iou_min;
    results[index].iou_max_sum = iou_max;
}
