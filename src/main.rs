use kiss3d::light::Light;
use kiss3d::nalgebra::{Isometry3, Matrix3, Point3, Translation3, UnitQuaternion, Vector3};
use kiss3d::window::Window;

use kdtree::KdTree;
use kdtree::distance::squared_euclidean;

use realsense_rust::{
    base::Rs2Intrinsics,
    config::Config,
    context::Context,
    frame::{ColorFrame, DepthFrame},
    kind::{Rs2Format, Rs2StreamKind},
    pipeline::InactivePipeline,
};

use realsense_rust::kind::{Rs2Extension, Rs2Option};

use std::collections::HashMap;
use std::error::Error;
use std::ffi::c_void;

const MODEL_PATH: &str = "/home/endre/rust_crates/realsense_interface/box_780.obj";
const DEPTH_WIDTH: usize = 640;
const DEPTH_HEIGHT: usize = 480;
const FPS: i32 = 30;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading CAD model from {}...", MODEL_PATH);
    let template_pcd = load_model_pcd(MODEL_PATH, 0.005)?;
    println!("Loaded {} points from model.", template_pcd.len());

    let context = Context::new()?;
    let devices = context.query_devices(std::collections::HashSet::new());
    let device = devices.into_iter().next().expect("No device found");

    let depth_sensor = device
        .sensors()
        .into_iter()
        .find(|s| s.extension() == Rs2Extension::DepthSensor)
        .expect("No depth sensor found");

    let depth_scale = depth_sensor
        .get_option_range(Rs2Option::DepthUnits)
        .unwrap()
        .default; 

    let inactive = InactivePipeline::try_from(&context)?;
    let mut cfg = Config::new();
    cfg.enable_stream(
        Rs2StreamKind::Depth,
        None,
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        Rs2Format::Z16,
        FPS as usize,
    )?;

    cfg.enable_stream(
        Rs2StreamKind::Color,
        None,
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        Rs2Format::Rgb8,
        FPS as usize,
    )?;

    // only this way to start pipeline (shadow inactive -> active)
    let mut pipeline = inactive.start(Some(cfg))?;
    let profile = pipeline.profile();

    let stream_profile = profile
        .streams()
        .into_iter()
        .find(|s| s.kind() == Rs2StreamKind::Depth)
        .expect("Depth stream not found in profile");

    let intrinsics: Rs2Intrinsics = stream_profile.intrinsics()?;

    let mut window = Window::new("RealSense PCD based pose estimation");
    window.set_light(Light::StickToCamera);

    while window.render() {
        let frames = match pipeline.wait(None) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Frame wait error: {:?}", e);
                continue;
            }
        };

        let depth_frame_opt: Option<DepthFrame> =
            frames.frames_of_type::<DepthFrame>().into_iter().next();
        let color_frame_opt: Option<ColorFrame> =
            frames.frames_of_type::<ColorFrame>().into_iter().next();

        if let (Some(depth_frame), Some(color_frame)) = (depth_frame_opt, color_frame_opt) {
            let (scene_pcd, scene_colors) =
                create_scene_pcd(&depth_frame, &color_frame, &intrinsics, depth_scale);

            let mut segmented_pcd: Vec<Point3<f32>> = scene_pcd
                .iter()
                .filter(|p| {
                    let dist = p.coords.magnitude();
                    dist > 0.2 && dist < 0.8
                })
                .cloned()
                .collect();

            let mut final_pose = Isometry3::identity();
            if segmented_pcd.len() > 100 {
                segmented_pcd = downsample_pcd(&segmented_pcd, 0.01);
                final_pose = simple_icp(&template_pcd, &segmented_pcd, 20, 0.005);
            }

            for (point, color) in scene_pcd.iter().zip(scene_colors.iter()) {
                window.draw_point(point, color);
            }
            for point in &segmented_pcd {
                window.draw_point(point, &Point3::new(0.0, 0.0, 1.0));
            }
            for point in &template_pcd {
                let transformed_point = final_pose * point;
                window.draw_point(&transformed_point, &Point3::new(0.0, 1.0, 0.0));
            }
        }
    }

    Ok(())
}

fn load_model_pcd(path: &str, sampling_density: f32) -> Result<Vec<Point3<f32>>, Box<dyn Error>> {
    let (models, _materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
    let mut pcd = Vec::new();

    for model in models.iter() {
        let mesh = &model.mesh;
        for v in 0..mesh.positions.len() / 3 {
            pcd.push(Point3::new(
                mesh.positions[3 * v],
                mesh.positions[3 * v + 1],
                mesh.positions[3 * v + 2],
            ));
        }
    }
    Ok(downsample_pcd(&pcd, sampling_density))
}

fn create_scene_pcd(
    depth_frame: &DepthFrame,
    color_frame: &ColorFrame,
    intrinsics: &Rs2Intrinsics,
    depth_scale: f32,
) -> (Vec<Point3<f32>>, Vec<Point3<f32>>) {
    let mut pcd = Vec::new();
    let mut colors = Vec::new();

    let d_w = depth_frame.width();
    let d_h = depth_frame.height();
    let c_w = color_frame.width();
    let c_h = color_frame.height();

    let depth_data: &[u16] = unsafe {
        let ptr = depth_frame.get_data() as *const c_void as *const u16;
        std::slice::from_raw_parts(ptr, d_w * d_h)
    };

    let color_data: &[u8] = unsafe {
        let ptr = color_frame.get_data() as *const c_void as *const u8;
        std::slice::from_raw_parts(ptr, c_w * c_h * 3)
    };

    for y in 0..d_h {
        for x in 0..d_w {
            let idx = y * d_w + x;
            let depth_val = depth_data[idx];
            if depth_val > 0 {
                let depth_m = depth_val as f32 * depth_scale;

                let point_x = (x as f32 - intrinsics.ppx()) / intrinsics.fx() * depth_m;
                let point_y = (y as f32 - intrinsics.ppy()) / intrinsics.fy() * depth_m;

                pcd.push(Point3::new(point_x, point_y, depth_m));

                let color_idx = (y * c_w + x) * 3;
                if color_idx + 2 < color_data.len() {
                    let r = color_data[color_idx] as f32 / 255.0;
                    let g = color_data[color_idx + 1] as f32 / 255.0;
                    let b = color_data[color_idx + 2] as f32 / 255.0;
                    colors.push(Point3::new(r, g, b));
                } else {
                    colors.push(Point3::new(0.5, 0.5, 0.5));
                }
            }
        }
    }
    (pcd, colors)
}

fn downsample_pcd(pcd: &[Point3<f32>], voxel_size: f32) -> Vec<Point3<f32>> {
    let mut voxel_sum: HashMap<(i32, i32, i32), (Vector3<f32>, usize)> = HashMap::new();
    for point in pcd {
        let vx = (point.x / voxel_size).floor() as i32;
        let vy = (point.y / voxel_size).floor() as i32;
        let vz = (point.z / voxel_size).floor() as i32;
        let key = (vx, vy, vz);
        let entry = voxel_sum.entry(key).or_insert((Vector3::zeros(), 0usize));
        entry.0 += point.coords;
        entry.1 += 1;
    }
    voxel_sum
        .into_iter()
        .map(|(_, (sum, count))| {
            let avg = sum / (count as f32);
            Point3::from(avg)
        })
        .collect()
}

fn simple_icp(
    source: &[Point3<f32>],
    target: &[Point3<f32>],
    max_iterations: usize,
    tolerance: f32,
) -> Isometry3<f32> {
    if source.is_empty() || target.is_empty() {
        return Isometry3::identity();
    }

    let mut target_kdtree: KdTree<f64, usize, [f64; 3]> = KdTree::new(3);
    for (i, point) in target.iter().enumerate() {
        target_kdtree
            .add([point.x as f64, point.y as f64, point.z as f64], i)
            .unwrap();
    }

    let mut current_source = source.to_vec();
    let mut total_transform = Isometry3::identity();

    for i in 0..max_iterations {
        let mut correspondences: Vec<(Point3<f32>, Point3<f32>)> =
            Vec::with_capacity(current_source.len());
        for s_point in &current_source {
            let nearest = target_kdtree
                .nearest(
                    &[s_point.x as f64, s_point.y as f64, s_point.z as f64],
                    1,
                    &squared_euclidean,
                )
                .unwrap();
            let nearest_idx = *nearest[0].1;
            let t_point = target[nearest_idx];
            correspondences.push((*s_point, t_point));
        }

        let n = correspondences.len() as f32;
        let sum_s = correspondences
            .iter()
            .map(|(s, _)| s.coords)
            .sum::<Vector3<f32>>();
        let sum_t = correspondences
            .iter()
            .map(|(_, t)| t.coords)
            .sum::<Vector3<f32>>();
        let source_centroid = sum_s / n;
        let target_centroid = sum_t / n;

        let mut h = Matrix3::zeros();
        for (s, t) in &correspondences {
            let s_centered = s.coords - source_centroid;
            let t_centered = t.coords - target_centroid;
            h += s_centered * t_centered.transpose();
        }

        let svd = h.svd(true, true);
        let u = svd.u.unwrap();
        let v_t = svd.v_t.unwrap();
        let mut r = v_t.transpose() * u.transpose();

        if r.determinant() < 0.0 {
            let mut v_t_corrected = v_t.clone();
            {
                let mut row = v_t_corrected.row_mut(2);
                row.scale_mut(-1.0);
            }
            r = v_t_corrected.transpose() * u.transpose();
        }

        let t_vec = target_centroid - r * source_centroid;
        let iteration_transform =
            Isometry3::from_parts(Translation3::from(t_vec), UnitQuaternion::from_matrix(&r));

        for s_point in &mut current_source {
            *s_point = iteration_transform * *s_point;
        }

        total_transform = iteration_transform * total_transform;

        let change = iteration_transform.translation.vector.norm()
            + (iteration_transform.rotation.angle() / std::f32::consts::PI);
        if i > 0 && change < tolerance {
            println!("ICP converged after {} iterations.", i);
            break;
        }
    }

    total_transform
}
