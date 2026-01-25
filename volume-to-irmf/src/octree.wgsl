struct Config {
    dims: vec3<u32>,
    level: u32,
    threshold_low: f32,
    threshold_high: f32,
    max_nodes: u32,
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var src_texture: texture_3d<f32>;
@group(0) @binding(2) var dst_texture: texture_storage_3d<r32float, write>;

// Pass 1: Mipmap generation (Reduction)
@compute @workgroup_size(8, 8, 1)
fn mipmap_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_coords = global_id;
    let src_coords = dst_coords * 2u;
    
    if (any(dst_coords >= textureDimensions(dst_texture))) {
        return;
    }

    var sum = 0.0;
    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let coords = src_coords + vec3<u32>(x, y, z);
                sum += textureLoad(src_texture, coords, 0).r;
            }
        }
    }
    
    textureStore(dst_texture, dst_coords, vec4f(sum / 8.0, 0.0, 0.0, 0.0));
}

struct Node {
    pos: vec3f,
    size: vec3f,
    occupancy: f32,
    _pad: f32,
}

@group(0) @binding(3) var<storage, read_write> out_nodes: array<Node>;
@group(0) @binding(4) var<storage, read_write> node_count: atomic<u32>;

// Pass 2: Node extraction
// Each workgroup checks a region at the current level.
@compute @workgroup_size(8, 8, 1)
fn extract_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if (any(global_id >= dims)) {
        return;
    }

    let occupancy = textureLoad(src_texture, global_id, 0).r;
    
    // If it's a leaf node (mostly full or we are at the target resolution)
    // threshold_high ~ 0.9, threshold_low ~ 0.1
    let is_full = occupancy >= config.threshold_high;
    let is_partial = occupancy > config.threshold_low && occupancy < config.threshold_high;
    let is_bottom_level = (config.level == 0u);

    if (is_full || (is_partial && is_bottom_level)) {
        let idx = atomicAdd(&node_count, 1u);
        if (idx < config.max_nodes) {
            let total_dims = vec3f(config.dims);
            let level_scale = f32(1u << config.level);
            
            let node_size = vec3f(level_scale) / total_dims / 2.0;
            let node_pos = (vec3f(global_id) * level_scale + vec3f(level_scale) / 2.0) / total_dims;
            
            out_nodes[idx].pos = node_pos;
            out_nodes[idx].size = node_size;
            out_nodes[idx].occupancy = occupancy;
        }
    }
}
