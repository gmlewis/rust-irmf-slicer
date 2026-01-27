struct Config {
    num_z: u32,
}

@group(0) @binding(0) var<storage, read> x1x2yz: array<vec4i>;
@group(0) @binding(1) var<storage, read> z_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> z_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4i>; // X1, X2, Y1, Y2
@group(0) @binding(4) var<storage, read_write> results_extra: array<i32>; // Z
@group(0) @binding(5) var<storage, read_write> result_count: atomic<u32>;
@group(0) @binding(6) var<uniform> config: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let z_idx = global_id.x;
    if (z_idx >= config.num_z) {
        return;
    }

    let offset = z_offsets[z_idx];
    let count = z_counts[z_idx];
    if (count == 0u) {
        return;
    }

    // X1X2YZ is sorted by X1 then Y.
    // We want to merge runs of Y where X1 and X2 are identical.
    
    var first = x1x2yz[offset];
    var x1 = first.x;
    var x2 = first.y;
    var y_start = first.z;
    var y_prev = y_start;
    let z = first.w;

    for (var i = 1u; i < count; i++) {
        let curr = x1x2yz[offset + i];
        if (curr.x == x1 && curr.y == x2 && curr.z == y_prev + 1) {
            y_prev = curr.z;
        } else {
            // End of run
            let res_idx = atomicAdd(&result_count, 1u);
            results[res_idx] = vec4i(x1, x2, y_start, y_prev);
            results_extra[res_idx] = z;
            
            x1 = curr.x;
            x2 = curr.y;
            y_start = curr.z;
            y_prev = y_start;
        }
    }
    // Last run
    let res_idx = atomicAdd(&result_count, 1u);
    results[res_idx] = vec4i(x1, x2, y_start, y_prev);
    results_extra[res_idx] = z;
}
