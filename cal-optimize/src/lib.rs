pub mod gpu_projector;

use irmf_slicer::{IrmfResult, Renderer, Slicer};
use ndarray::{Array3, ArrayBase, Data, Ix3};

/// A target volume generated from an IRMF model.
#[derive(Clone)]
pub struct TargetVolume {
    pub data: Array3<f32>,
    pub res_microns: [f32; 3],
}

impl TargetVolume {
    /// Creates a new TargetVolume from an IRMF model by slicing it along the Z axis.
    pub fn from_irmf<R: Renderer>(slicer: &mut Slicer<R>, material_num: usize) -> IrmfResult<Self> {
        Self::from_irmf_with_callback(slicer, material_num, || {})
    }

    pub fn from_irmf_with_callback<R: Renderer, F: Fn()>(
        slicer: &mut Slicer<R>,
        material_num: usize,
        callback: F,
    ) -> IrmfResult<Self> {
        // println!("Final Shader:\n{}", slicer.model.shader);
        slicer.prepare_render_z()?;
        let nz = slicer.num_z_slices();
        let ny = slicer.num_y_slices();
        let nx = slicer.num_x_slices();

        let mut data = Array3::zeros((nx, ny, nz));

        slicer.render_z_slices(material_num, |z_idx, _z_depth, _radius, img| {
            let luma = img.to_luma32f();
            let mut slice_max = 0.0f32;
            // let mut slice_sum = 0.0f32;
            for (pixel_x, pixel_y, pixel) in luma.enumerate_pixels() {
                let val = pixel.0[0];
                if val > slice_max {
                    slice_max = val;
                }
                // slice_sum += val;
                if (pixel_x as usize) < nx && (pixel_y as usize) < ny {
                    data[[pixel_x as usize, pixel_y as usize, z_idx]] = val;
                }
            }
            // println!("Target generation: slice {} at depth {:.4}, max: {:.4}, sum: {:.4}", z_idx, z_depth, slice_max, slice_sum);
            callback();
            Ok(())
        })?;

        Ok(Self {
            data,
            res_microns: [slicer.res_x, slicer.res_y, slicer.res_z],
        })
    }
}

pub trait Projector {
    fn forward(&self, volume: &Array3<f32>, angles: &[f32]) -> Array3<f32>;
    fn backward(
        &self,
        projections: &Array3<f32>,
        angles: &[f32],
        target_dim: (usize, usize, usize),
    ) -> Array3<f32>;
}

pub struct CpuProjector;

impl Projector for CpuProjector {
    fn forward(&self, volume: &Array3<f32>, angles: &[f32]) -> Array3<f32> {
        let (nx, ny, nz) = volume.dim();
        let n_angles = angles.len();
        let nr = ((nx as f32).powi(2) + (ny as f32).powi(2)).sqrt().ceil() as usize;
        let mut projections = Array3::zeros((nr, n_angles, nz));

        let center_x = (nx as f32 - 1.0) / 2.0;
        let center_y = (ny as f32 - 1.0) / 2.0;
        let center_r = (nr as f32 - 1.0) / 2.0;

        for z in 0..nz {
            for (a_idx, &angle_deg) in angles.iter().enumerate() {
                let angle_rad = angle_deg.to_radians();
                let cos_a = angle_rad.cos();
                let sin_a = angle_rad.sin();

                for r_idx in 0..nr {
                    let r = r_idx as f32 - center_r;
                    let mut sum = 0.0;
                    for t_idx in 0..nr {
                        let t = t_idx as f32 - center_r;
                        let x = r * cos_a - t * sin_a + center_x;
                        let y = r * sin_a + t * cos_a + center_y;

                        if x >= 0.0 && x < (nx - 1) as f32 && y >= 0.0 && y < (ny - 1) as f32 {
                            let x0 = x.floor() as usize;
                            let y0 = y.floor() as usize;
                            let x1 = x0 + 1;
                            let y1 = y0 + 1;
                            let dx = x - x0 as f32;
                            let dy = y - y0 as f32;

                            let val = (1.0 - dx) * (1.0 - dy) * volume[[x0, y0, z]]
                                + dx * (1.0 - dy) * volume[[x1, y0, z]]
                                + (1.0 - dx) * dy * volume[[x0, y1, z]]
                                + dx * dy * volume[[x1, y1, z]];
                            sum += val;
                        }
                    }
                    projections[[r_idx, a_idx, z]] = sum;
                }
            }
        }
        projections
    }

    fn backward(
        &self,
        projections: &Array3<f32>,
        angles: &[f32],
        target_dim: (usize, usize, usize),
    ) -> Array3<f32> {
        let (nr, n_angles, nz) = projections.dim();
        let (nx, ny, _nz_target) = target_dim;
        let mut volume = Array3::zeros((nx, ny, nz));

        let center_x = (nx as f32 - 1.0) / 2.0;
        let center_y = (ny as f32 - 1.0) / 2.0;
        let center_r = (nr as f32 - 1.0) / 2.0;

        for z in 0..nz {
            for (a_idx, &angle_deg) in angles.iter().enumerate() {
                let angle_rad = angle_deg.to_radians();
                let cos_a = angle_rad.cos();
                let sin_a = angle_rad.sin();

                for x_idx in 0..nx {
                    let x = x_idx as f32 - center_x;
                    for y_idx in 0..ny {
                        let y = y_idx as f32 - center_y;
                        let r = x * cos_a + y * sin_a + center_r;

                        if r >= 0.0 && r < (nr - 1) as f32 {
                            let r0 = r.floor() as usize;
                            let r1 = r0 + 1;
                            let dr = r - r0 as f32;
                            let val = (1.0 - dr) * projections[[r0, a_idx, z]]
                                + dr * projections[[r1, a_idx, z]];
                            volume[[x_idx, y_idx, z]] += val;
                        }
                    }
                }
            }
        }
        volume /= n_angles as f32;
        volume
    }
}

pub struct CalOptimizer<P: Projector> {
    pub target: TargetVolume,
    pub angles: Vec<f32>,
    pub learning_rate: f32,
    pub sigmoid_param: f32,
    pub max_iter: usize,
    pub projector: P,
    pub use_filter: bool,
}

impl<P: Projector> CalOptimizer<P> {
    pub fn new(target: TargetVolume, angles: Vec<f32>, projector: P) -> Self {
        Self {
            target,
            angles,
            learning_rate: 0.005,
            sigmoid_param: 0.01,
            max_iter: 10,
            projector,
            use_filter: true,
        }
    }

    pub fn find_threshold<D: Data<Elem = f32>>(&self, x: &ArrayBase<D, Ix3>) -> f32 {
        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;
        for &val in x.iter() {
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        let num_tests = 100;
        let mut best_thresh = min_val;
        let mut best_score = f32::MIN;

        let gel_indices: Vec<_> = self
            .target
            .data
            .indexed_iter()
            .filter(|&(_, &v)| v > 0.5)
            .map(|(idx, _)| idx)
            .collect();
        let void_indices: Vec<_> = self
            .target
            .data
            .indexed_iter()
            .filter(|&(_, &v)| v <= 0.5)
            .map(|(idx, _)| idx)
            .collect();

        if gel_indices.is_empty() || void_indices.is_empty() {
            return (min_val + max_val) / 2.0;
        }

        for i in 0..num_tests {
            let thresh = min_val + (max_val - min_val) * (i as f32) / (num_tests as f32);
            let mut gel_in_target = 0;
            for &idx in &gel_indices {
                if x[idx] >= thresh {
                    gel_in_target += 1;
                }
            }
            let mut gel_not_in_target = 0;
            for &idx in &void_indices {
                if x[idx] >= thresh {
                    gel_not_in_target += 1;
                }
            }

            let score = (gel_in_target as f32 / gel_indices.len() as f32)
                - (gel_not_in_target as f32 / void_indices.len() as f32);
            if score > best_score {
                best_score = score;
                best_thresh = thresh;
            }
        }
        best_thresh
    }

    pub fn calc_ver<D: Data<Elem = f32>>(&self, recon: &ArrayBase<D, Ix3>) -> f32 {
        let mut min_gel_dose = f32::MAX;
        let mut gel_count = 0;
        for (idx, &t_val) in self.target.data.indexed_iter() {
            if t_val > 0.5 {
                let r_val = recon[idx];
                if r_val < min_gel_dose {
                    min_gel_dose = r_val;
                }
                gel_count += 1;
            }
        }

        let mut n_pix_overlap = 0;
        let mut void_count = 0;
        for (idx, &t_val) in self.target.data.indexed_iter() {
            if t_val <= 0.5 {
                if recon[idx] >= min_gel_dose {
                    n_pix_overlap += 1;
                }
                void_count += 1;
            }
        }

        if gel_count + void_count == 0 {
            return 0.0;
        }
        n_pix_overlap as f32 / (gel_count + void_count) as f32
    }

    pub fn run(&mut self) -> (Array3<f32>, Vec<f32>) {
        self.run_with_callback(|_, _| {})
    }

    pub fn run_with_callback<F: Fn(usize, f32)>(&mut self, callback: F) -> (Array3<f32>, Vec<f32>) {
        let mut b = self.projector.forward(&self.target.data, &self.angles);

        if self.use_filter {
            apply_ram_lak(&mut b);
            for val in b.iter_mut() {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }

        let mut opt_b = b.clone();
        let mut errors = Vec::new();

        for iter in 0..self.max_iter {
            let mut x = self
                .projector
                .backward(&opt_b, &self.angles, self.target.data.dim());
            let max_x = x.iter().fold(0.0f32, |m, &v| m.max(v));
            if max_x > 0.0 {
                x /= max_x;
            }

            let ver = self.calc_ver(&x);
            errors.push(ver);
            callback(iter + 1, ver);

            let mu = self.find_threshold(&x);
            let mut x_thresh = x.clone();
            for val in x_thresh.iter_mut() {
                *val = sigmoid(*val - mu, self.sigmoid_param);
            }

            let delta_x = x_thresh - &self.target.data;
            let delta_b = self.projector.forward(&delta_x, &self.angles);
            opt_b = opt_b - (&delta_b * self.learning_rate);
            for val in opt_b.iter_mut() {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
        }

        (opt_b, errors)
    }
}

use rustfft::{FftPlanner, num_complex::Complex};

pub fn apply_ram_lak(projections: &mut Array3<f32>) {
    let (nr, n_angles, nz) = projections.dim();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(nr);
    let ifft = planner.plan_fft_inverse(nr);

    // Create the ramp filter in frequency domain
    let mut filter = vec![0.0f32; nr];
    let center = nr as f32 / 2.0;
    for (i, val) in filter.iter_mut().enumerate() {
        let f = (i as f32 - center) / nr as f32;
        *val = f.abs();
    }
    // Shift filter for FFT
    filter.rotate_right(nr / 2);

    for z in 0..nz {
        for a in 0..n_angles {
            let mut line: Vec<Complex<f32>> = (0..nr)
                .map(|r| Complex::new(projections[[r, a, z]], 0.0))
                .collect();

            fft.process(&mut line);
            for (i, val) in line.iter_mut().enumerate() {
                *val *= filter[i];
            }
            ifft.process(&mut line);

            for r in 0..nr {
                projections[[r, a, z]] = line[r].re / nr as f32;
            }
        }
    }
}

pub fn sigmoid(x: f32, g: f32) -> f32 {
    1.0 / (1.0 + (-x / g).exp())
}
