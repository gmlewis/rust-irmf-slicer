struct Config {
    dims: vec4u, // Explicit padding
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
    let dst_dims = textureDimensions(dst_texture);
    if (any(global_id >= dst_dims)) {
        return;
    }

    let src_coords = global_id * 2u;
    let src_dims = textureDimensions(src_texture);

    var sum = 0.0;
    var count = 0.0;
    for (var z = 0u; z < 2u; z++) {
        for (var y = 0u; y < 2u; y++) {
            for (var x = 0u; x < 2u; x++) {
                let coords = src_coords + vec3<u32>(x, y, z);
                if (all(coords < src_dims)) {
                    sum += textureLoad(src_texture, coords, 0).r;
                    count += 1.0;
                }
            }
        }
    }
    
    // Average occupancy in this block
    let avg = select(0.0, sum / count, count > 0.0);
    textureStore(dst_texture, global_id, vec4f(avg, 0.0, 0.0, 0.0));
}

struct Node {
    pos: vec4f,
    size: vec4f,
    data: vec4f, // x: occupancy, y: level
}

@group(0) @binding(3) var<storage, read_write> out_nodes: array<Node>;
@group(0) @binding(4) var<storage, read_write> node_count: atomic<u32>;

// Pass 2: Node extraction
@compute @workgroup_size(8, 8, 1)
fn extract_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(src_texture);
    if (any(global_id >= dims)) {
        return;
    }

    let occupancy = textureLoad(src_texture, global_id, 0).r;
    
    // Low thresholds ensure we don't miss sparse features like the Rodin coil wires.
    let is_leaf = (occupancy >= config.threshold_high) || (config.level == 0u && occupancy > config.threshold_low);

    if (is_leaf) {
        let idx = atomicAdd(&node_count, 1u);
        if (idx < config.max_nodes) {
            let total_dims = vec3f(config.dims.xyz);
            let level_scale = f32(1u << config.level);
            
            // Half-size (radius) of the node in normalized [0, 1] space
            let node_size = (vec3f(level_scale) / total_dims) / 2.0;
            // Center position of the node in normalized [0, 1] space
            let node_pos = (vec3f(global_id) * level_scale + vec3f(level_scale) / 2.0) / total_dims;
            
            out_nodes[idx].pos = vec4f(node_pos, 0.0);
            out_nodes[idx].size = vec4f(node_size, 0.0);
            out_nodes[idx].data = vec4f(occupancy, f32(config.level), 0.0, 0.0);
        }
    }
}