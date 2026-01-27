struct Config {
    num_yz: u32,
    dims_y: u32,
    dims_z: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> x_indices: array<i32>;
@group(0) @binding(1) var<storage, read> yz_offsets: array<u32>;
@group(0) @binding(2) var<storage, read> yz_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> results: array<vec4i>;
@group(0) @binding(4) var<storage, read_write> result_count: atomic<u32>;
@group(0) @binding(5) var<uniform> config: Config;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let yz_idx = global_id.x;
    if (yz_idx >= config.num_yz) {
        return;
    }

    let offset = yz_offsets[yz_idx];
    let count = yz_counts[yz_idx];
    if (count == 0u) {
        return;
    }

    let y = i32(yz_idx % config.dims_y);
    let z = i32(yz_idx / config.dims_y);

    var x_start = x_indices[offset];
    var x_prev = x_start;

    for (var i = 1u; i < count; i++) {
        let x_curr = x_indices[offset + i];
        if (x_curr != x_prev + 1) {
            // End of run
            let res_idx = atomicAdd(&result_count, 1u);
            results[res_idx] = vec4i(x_start, x_prev, y, z);
            x_start = x_curr;
        }
        x_prev = x_curr;
    }
    // Last run
    let res_idx = atomicAdd(&result_count, 1u);
    results[res_idx] = vec4i(x_start, x_prev, y, z);
}
