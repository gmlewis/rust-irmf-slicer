use irmf_slicer::{Slicer, Renderer, IrmfResult};
use std::fs::File;
use std::io::{Write, Seek, SeekFrom};
use image::{RgbaImage};

const SCREEN_WIDTH: u32 = 0xa00;
const SCREEN_HEIGHT: u32 = 0x5a0;
const PREVIEW_WIDTH: u32 = 0x190;
const PREVIEW_HEIGHT: u32 = 0x12c;
const THUMBNAIL_WIDTH: u32 = 0xc8;
const THUMBNAIL_HEIGHT: u32 = 0x7d;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FileHeader {
    magic1: u32,
    magic2: u32,
    plate_x: f32,
    plate_y: f32,
    plate_z: f32,
    field_14: u32,
    field_18: u32,
    field_1c: u32,
    layer_thickness: f32,
    normal_exposure_time: f32,
    bottom_exposure_time: f32,
    off_time: f32,
    bottom_layers: u32,
    screen_height: u32,
    screen_width: u32,
    preview_header_offset: u32,
    layer_headers_offset: u32,
    total_layers: u32,
    preview_thumbnail_header_offset: u32,
    field_4c: u32,
    light_curing_type: u32,
    field_54: u32,
    field_58: u32,
    field_60: u32,
    field_5c: u32,
    field_64: u32,
    field_68: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PreviewHeader {
    width: u32,
    height: u32,
    preview_data_offset: u32,
    preview_data_size: u32,
    field_10: u32,
    field_14: u32,
    field_18: u32,
    field_1c: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerHeader {
    absolute_height: f32,
    exposure_time: f32,
    per_layer_off_time: f32,
    image_data_offset: u32,
    image_data_size: u32,
    field_14: u32,
    field_18: u32,
    field_1c: u32,
    field_20: u32,
}

pub fn slice_to_photon<R: Renderer>(slicer: &mut Slicer<R>, material_num: usize, filename: &str, z_res: f32) -> IrmfResult<()> {
    let mut file = File::create(filename).map_err(|e| anyhow::anyhow!("File::create: {}", e))?;
    let num_slices = slicer.num_z_slices();

    println!("Rendering Z-slices for Photon...");
    slicer.prepare_render_z().map_err(|e| anyhow::anyhow!("prepare_render_z: {}", e))?;

    let mut layer_headers = Vec::with_capacity(num_slices);
    let mut current_pos = 0u32;
    
    let mut header = FileHeader {
        magic1: 0x12FD0019,
        magic2: 0x01,
        plate_x: 68.04,
        plate_y: 120.96,
        plate_z: 150.0,
        field_14: 0, field_18: 0, field_1c: 0,
        layer_thickness: z_res / 1000.0,
        normal_exposure_time: 6.0,
        bottom_exposure_time: 50.0,
        off_time: 0.0,
        bottom_layers: 8,
        screen_height: SCREEN_HEIGHT,
        screen_width: SCREEN_WIDTH,
        preview_header_offset: 0,
        layer_headers_offset: 0,
        total_layers: num_slices as u32,
        preview_thumbnail_header_offset: 0,
        field_4c: 0,
        light_curing_type: 1,
        field_54: 0, field_58: 0, field_60: 0, field_5c: 0, field_64: 0, field_68: 0,
    };
    
    file.write_all(bytemuck::bytes_of(&header))?;
    current_pos += std::mem::size_of::<FileHeader>() as u32;

    let first_img = slicer.render_z_slice(0, material_num)?;
    let rgba = first_img.to_rgba8();
    
    let preview_data = encode_preview(PREVIEW_WIDTH, PREVIEW_HEIGHT, &rgba);
    let thumbnail_data = encode_preview(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT, &rgba);
    
    header.preview_header_offset = current_pos;
    let preview_header = PreviewHeader {
        width: PREVIEW_WIDTH,
        height: PREVIEW_HEIGHT,
        preview_data_offset: current_pos + std::mem::size_of::<PreviewHeader>() as u32,
        preview_data_size: preview_data.len() as u32,
        field_10: 0, field_14: 0, field_18: 0, field_1c: 0,
    };
    file.write_all(bytemuck::bytes_of(&preview_header))?;
    file.write_all(&preview_data)?;
    current_pos += (std::mem::size_of::<PreviewHeader>() + preview_data.len()) as u32;
    
    header.preview_thumbnail_header_offset = current_pos;
    let thumbnail_header = PreviewHeader {
        width: THUMBNAIL_WIDTH,
        height: THUMBNAIL_HEIGHT,
        preview_data_offset: current_pos + std::mem::size_of::<PreviewHeader>() as u32,
        preview_data_size: thumbnail_data.len() as u32,
        field_10: 0, field_14: 0, field_18: 0, field_1c: 0,
    };
    file.write_all(bytemuck::bytes_of(&thumbnail_header))?;
    file.write_all(&thumbnail_data)?;
    current_pos += (std::mem::size_of::<PreviewHeader>() + thumbnail_data.len()) as u32;
    
    header.layer_headers_offset = current_pos;
    let layer_headers_start = current_pos;
    for _ in 0..num_slices {
        file.write_all(bytemuck::bytes_of(&LayerHeader {
            absolute_height: 0.0, exposure_time: 0.0, per_layer_off_time: 0.0,
            image_data_offset: 0, image_data_size: 0, field_14: 0, field_18: 0, field_1c: 0, field_20: 0,
        }))?;
        current_pos += std::mem::size_of::<LayerHeader>() as u32;
    }
    
    for i in 0..num_slices {
        let img = if i == 0 { first_img.clone() } else { slicer.render_z_slice(i, material_num)? };
        let layer_data = encode_layer_image_data(&img.to_rgba8());
        
        let exp_time = if i < header.bottom_layers as usize { header.bottom_exposure_time } else { header.normal_exposure_time };
        layer_headers.push(LayerHeader {
            absolute_height: (i as f32) * z_res / 1000.0,
            exposure_time: exp_time,
            per_layer_off_time: header.off_time,
            image_data_offset: current_pos,
            image_data_size: layer_data.len() as u32,
            field_14: 0, field_18: 0, field_1c: 0, field_20: 0,
        });
        
        file.write_all(&layer_data)?;
        current_pos += layer_data.len() as u32;
    }
    
    file.seek(SeekFrom::Start(0))?;
    file.write_all(bytemuck::bytes_of(&header))?;
    
    file.seek(SeekFrom::Start(layer_headers_start as u64))?;
    for lh in layer_headers {
        file.write_all(bytemuck::bytes_of(&lh))?;
    }

    Ok(())
}

fn encode_layer_image_data(img: &RgbaImage) -> Vec<u8> {
    const FLAG_SET_PIXELS: u8 = 0x80;
    let mut output = Vec::new();

    let x_offset = if img.width() < SCREEN_WIDTH { (SCREEN_WIDTH - img.width()) / 2 } else { 0 };
    let y_offset = if img.height() < SCREEN_HEIGHT { (SCREEN_HEIGHT - img.height()) / 2 } else { 0 };

    let mut unset_count = 0u8;
    let mut set_count = 0u8;

    for pixel_index in 0..(SCREEN_WIDTH * SCREEN_HEIGHT) {
        let y = pixel_index % SCREEN_HEIGHT;
        let x = pixel_index / SCREEN_HEIGHT;

        let pixel_on = if x >= x_offset && x < x_offset + img.width() && y >= y_offset && y < y_offset + img.height() {
            img.get_pixel(x - x_offset, y - y_offset)[0] > 0
        } else {
            false
        };

        if !pixel_on {
            if set_count != 0 {
                output.push(set_count | FLAG_SET_PIXELS);
                set_count = 0;
            }
            unset_count += 1;
            if unset_count >= 0x7f - 2 {
                output.push(unset_count);
                unset_count = 0;
            }
        } else {
            if unset_count != 0 {
                output.push(unset_count);
                unset_count = 0;
            }
            set_count += 1;
            if set_count >= 0x7f - 2 {
                output.push(set_count | FLAG_SET_PIXELS);
                set_count = 0;
            }
        }
    }

    if set_count != 0 { output.push(set_count | FLAG_SET_PIXELS); }
    if unset_count != 0 { output.push(unset_count); }

    output
}

fn encode_preview(image_width: u32, image_height: u32, img: &RgbaImage) -> Vec<u8> {
    let mut output = Vec::new();
    let x_scale = (img.width() as f32) / (image_width as f32);
    let y_scale = (img.height() as f32) / (image_height as f32);

    let pixel_at = |pi: u32| {
        let x = pi % image_width;
        let y = pi / image_width;
        let nx = (x as f32 * x_scale) as u32;
        let ny = (y as f32 * y_scale) as u32;
        if nx < img.width() && ny < img.height() {
            *img.get_pixel(nx, ny)
        } else {
            image::Rgba([0, 0, 0, 0])
        }
    };

    let max_pixel_index = image_height * image_width;
    let mut pi = 0u32;
    while pi < max_pixel_index {
        let p = pixel_at(pi);
        
        let mut skip_count = 1u32;
        while pi + skip_count < max_pixel_index && skip_count < 0xFFF && pixel_at(pi + skip_count) == p {
            skip_count += 1;
        }

        if skip_count < 3 {
            let v = combine_rgb5515(p[0], p[1], p[2], false);
            output.extend_from_slice(&v.to_le_bytes());
            pi += 1;
        } else {
            let v = combine_rgb5515(p[0], p[1], p[2], true) | 0x20;
            output.extend_from_slice(&v.to_le_bytes());
            let skip_v = ((skip_count as u16) - 1) | 0x3000;
            output.extend_from_slice(&skip_v.to_le_bytes());
            pi += skip_count;
        }
    }
    output
}

fn combine_rgb5515(r: u8, g: u8, b: u8, is_fill: bool) -> u16 {
    let r_bits = (r as u32 * 31 / 255) as u16;
    let g_bits = (g as u32 * 31 / 255) as u16;
    let b_bits = (b as u32 * 31 / 255) as u16;
    let fill_bit = if is_fill { 1u16 } else { 0u16 };

    (r_bits & 0x1F) | ((fill_bit & 0x1) << 5) | ((g_bits & 0x1F) << 6) | ((b_bits & 0x1F) << 11)
}