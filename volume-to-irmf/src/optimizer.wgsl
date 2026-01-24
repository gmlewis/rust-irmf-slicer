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
    sample_idx: u32,
}

@group(0) @binding(0) var<storage, read> primitives: array<Primitive>;
@group(0) @binding(1) var target_volume: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> results: array<ErrorResult>;

struct Config {
    num_samples: u32,
    num_primitives: u32,
    seed: u32,
    num_candidates: u32,
}
@group(0) @binding(3) var<uniform> config: Config;

struct Perturbation {
    prim_idx: u32,
    pos_delta: vec3f,
    size_scale: f32,
    op: u32,
}
@group(0) @binding(4) var<storage, read> perturbations: array<Perturbation>;

@group(0) @binding(5) var<storage, read> samples: array<vec3f>;

var<workgroup> local_mse: array<f32, 256>;
var<workgroup> local_min: array<f32, 256>;
var<workgroup> local_max: array<f32, 256>;

fn sd_sphere(p: vec3f, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sd_box(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn evaluate_model(p: vec3f, cand_idx: u32) -> f32 {
    var val = 0.0;
    let pert = perturbations[cand_idx];
    
    let is_adding = (pert.prim_idx == 9999u);
    let num_prims = select(config.num_primitives, config.num_primitives + 1u, is_adding);

    for (var i = 0u; i < num_prims; i++) {
        var prim: Primitive;
        if (i < config.num_primitives) {
            prim = primitives[i];
            if (i == pert.prim_idx) {
                prim.pos += pert.pos_delta;
                prim.size *= pert.size_scale;
            }
        } else {
            prim.pos = pert.pos_delta;
            prim.size = vec3f(pert.size_scale);
            prim.prim_type = select(0u, 1u, pert.op >= 2u);
            prim.op = pert.op % 2u;
        }

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

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let cand_idx = group_id.x;
    let group_in_cand = group_id.y;
    let sample_idx = group_in_cand * 256u + local_id.x;
    
    if (cand_idx >= config.num_candidates || sample_idx >= config.num_samples) {
        return;
    }

    let uvw = samples[sample_idx];
    let dims = textureDimensions(target_volume);
    let coords = vec3<u32>(uvw * vec3f(dims));
    let target_val = textureLoad(target_volume, coords, 0).r;
    let candidate_val = evaluate_model(uvw, cand_idx);

    let diff = target_val - candidate_val;
    local_mse[local_id.x] = diff * diff;
    local_min[local_id.x] = min(target_val, candidate_val);
    local_max[local_id.x] = max(target_val, candidate_val);

    workgroupBarrier();

    // Reduction
    if (local_id.x == 0u) {
        var mse_sum = 0.0;
        var min_sum = 0.0;
        var max_sum = 0.0;
        for (var i = 0u; i < 256u; i++) {
            mse_sum += local_mse[i];
            min_sum += local_min[i];
            max_sum += local_max[i];
        }
        
        // Use group_id.y to store partial sums
        let out_idx = cand_idx * (config.num_samples / 256u) + group_in_cand;
        results[out_idx].mse_sum = mse_sum;
        results[out_idx].iou_min_sum = min_sum;
        results[out_idx].iou_max_sum = max_sum;
    }
}