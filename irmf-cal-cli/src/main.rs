use anyhow::{Result, anyhow};
use cal_hardware::{
    MockMotor, Motor, ProjectionController, ThorlabsMotor, projection_window::ProjectionWindow,
};
use cal_optimize::{CalOptimizer, CpuProjector, TargetVolume, gpu_projector::GpuProjector};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use irmf_slicer::{Slicer, irmf::IrmfModel, wgpu_renderer::WgpuRenderer};
use ndarray::Array3;
use std::path::PathBuf;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the .irmf file
    #[arg(short, long)]
    input: PathBuf,

    /// Resolution in microns
    #[arg(short, long, default_value_t = 1000.0)]
    res: f32,

    /// Force CPU optimization instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Number of optimization iterations
    #[arg(long, default_value_t = 10)]
    iterations: usize,

    /// Index of the monitor to use for projection
    #[arg(long)]
    monitor: Option<usize>,

    /// Disable tomographic filtering (Ram-Lak)
    #[arg(long)]
    no_filter: bool,

    /// Use a mock motor instead of physical hardware
    #[arg(long)]
    mock: bool,

    /// Serial port for the Thorlabs motor
    #[arg(long)]
    port: Option<String>,

    /// Number of full rotations to perform during printing
    #[arg(long, default_value_t = 1)]
    rotations: usize,
}

struct CalApp<M: Motor + Send + 'static> {
    args: Args,
    opt_projections: Option<Array3<f32>>,
    _angles: Vec<f32>,
    window: Option<ProjectionWindow>,
    controller: Option<ProjectionController<M>>,
    last_frame_idx: usize,
    print_pb: Option<ProgressBar>,
    stop_signal: Arc<std::sync::atomic::AtomicBool>,
}

impl<M: Motor + Send + 'static> ApplicationHandler for CalApp<M> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window at a visible size
        let window = pollster::block_on(ProjectionWindow::new(
            event_loop,
            512,
            512,
            self.args.monitor,
        ))
        .expect("Failed to create projection window");

        // Prepare textures
        let mut window = window;
        let projections = self.opt_projections.as_ref().unwrap();
        window.prepare_projections(projections);

        self.window = Some(window);

        // Start motor and sync thread
        if let Some(ref mut controller) = self.controller {
            println!(
                "Starting motor rotation ({} rotations)...",
                self.args.rotations
            );
            let mut m = controller.motor.lock().unwrap();
            m.start_rotation(10.0).expect("Failed to start motor");
            drop(m);
            controller.spawn_sync_thread(Arc::clone(&self.stop_signal));
        }

        let total_frames = self.args.rotations * self._angles.len();
        let pb = ProgressBar::new(total_frames as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.magenta/red}] {pos}/{len} frames (Rotations: {msg})")
            .unwrap());
        self.print_pb = Some(pb);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested | WindowEvent::KeyboardInput { .. } => {
                println!("\nExiting early...");
                self.cleanup();
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let (Some(window), Some(controller)) = (&self.window, &mut self.controller) {
                    controller.poll_and_render(window).unwrap();
                    let current_idx = controller
                        .shared_frame_index()
                        .load(std::sync::atomic::Ordering::Relaxed);

                    if current_idx != self.last_frame_idx {
                        if let Some(ref pb) = self.print_pb {
                            pb.inc(1);
                            let current_rotations =
                                pb.position() as f32 / self._angles.len() as f32;
                            pb.set_message(format!(
                                "{:.2}/{}",
                                current_rotations, self.args.rotations
                            ));

                            if current_rotations >= self.args.rotations as f32 {
                                pb.finish_with_message("Print complete");
                                self.cleanup();
                                event_loop.exit();
                            }
                        }
                        self.last_frame_idx = current_idx;
                    }
                }
                // Continuously redraw
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

impl<M: Motor + Send + 'static> CalApp<M> {
    fn cleanup(&mut self) {
        self.stop_signal
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(ref mut controller) = self.controller {
            if let Ok(mut m) = controller.motor.lock() {
                let _ = m.stop();
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading IRMF model: {:?}", args.input);
    let data = std::fs::read(&args.input)?;
    let model = IrmfModel::new(&data)?;
    println!(
        "IRMF Model bounds: min {:?}, max {:?}",
        model.header.min, model.header.max
    );

    let renderer = WgpuRenderer::new().await?;
    let mut slicer = Slicer::new(model, renderer, args.res, args.res, args.res);

    println!("Generating target volume...");
    let mut target = None;
    let num_materials = slicer.model.header.materials.len();

    for m_idx in 0..num_materials {
        println!(
            "Trying material {} ({:?})...",
            m_idx, slicer.model.header.materials[m_idx]
        );
        let nz = slicer.num_z_slices();
        let pb = ProgressBar::new(nz as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} slices ({eta})")?
            .progress_chars("#>-"));

        let t = TargetVolume::from_irmf_with_callback(&mut slicer, m_idx + 1, || {
            pb.inc(1);
        })?;

        let vol_max = t.data.iter().fold(0.0f32, |m, &v| m.max(v));
        if vol_max > 0.0 {
            pb.finish_with_message(format!("Target volume generated using material {}", m_idx));
            target = Some(t);
            break;
        } else {
            pb.finish_with_message(format!("Material {} produced an empty volume", m_idx));
        }
    }

    let target = target.ok_or_else(|| {
        anyhow!("All materials produced empty volumes. Check your IRMF shader logic.")
    })?;
    println!("Target volume dimensions: {:?}", target.data.dim());

    let angles: Vec<f32> = (0..180).map(|a| a as f32).collect();

    let pb_opt = ProgressBar::new(args.iterations as u64);
    pb_opt.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/red}] {pos}/{len} iterations (VER: {msg})")?);

    let (opt_b, errors) = if args.cpu {
        println!("Starting CPU optimization...");
        let mut optimizer = CalOptimizer::new(target, angles.clone(), CpuProjector);
        optimizer.max_iter = args.iterations;
        optimizer.use_filter = !args.no_filter;
        optimizer.run_with_callback(|iter, ver| {
            pb_opt.set_position(iter as u64);
            pb_opt.set_message(format!("{:.4}", ver));
        })
    } else {
        println!("Starting GPU optimization...");
        let projector = GpuProjector::new().await;
        let mut optimizer = CalOptimizer::new(target, angles.clone(), projector);
        optimizer.max_iter = args.iterations;
        optimizer.use_filter = !args.no_filter;
        optimizer.run_with_callback(|iter, ver| {
            pb_opt.set_position(iter as u64);
            pb_opt.set_message(format!("{:.4}", ver));
        })
    };
    pb_opt.finish_with_message("Optimization complete");

    println!("Optimization finished. Final VER: {:?}", errors.last());
    println!("Projections ready. Starting hardware sync loop...");

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let stop_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));

    if args.mock {
        let motor = MockMotor::new();
        let controller = ProjectionController::new(motor, opt_b.clone(), angles.clone());
        let mut app = CalApp {
            args,
            opt_projections: Some(opt_b),
            _angles: angles,
            window: None,
            controller: Some(controller),
            last_frame_idx: 999,
            print_pb: None,
            stop_signal,
        };
        event_loop.run_app(&mut app)?;
    } else {
        let port = args
            .port
            .as_ref()
            .ok_or_else(|| anyhow!("--port required for physical motor"))?;
        let motor = ThorlabsMotor::new(port)?;
        let controller = ProjectionController::new(motor, opt_b.clone(), angles.clone());
        let mut app = CalApp {
            args,
            opt_projections: Some(opt_b),
            _angles: angles,
            window: None,
            controller: Some(controller),
            last_frame_idx: 999,
            print_pb: None,
            stop_signal,
        };
        event_loop.run_app(&mut app)?;
    }

    Ok(())
}
