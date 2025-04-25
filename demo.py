import argparse

from depth_predictor import DepthPredictor
from pose_rollout_predictor import PoseRolloutPredictor
from utils import *
from visualizer import PoseTrajectoryVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Demo for RollingDepth depth prediction"
    )
    # parser.add_argument(
    #     '-i', '--input-video', type=str, required=True,
    #     help='Path to the input video file'
    # )
    # parser.add_argument(
    #     '-o', '--output-dir', type=str, required=True,
    #     help='Directory to save prediction outputs'
    # )
    parser.add_argument(
        "-i",
        "--input-video",
        type=str,
        default="outputs/generated_video.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save prediction outputs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/rollingdepth-v1-0",
        help="Model checkpoint (local path or HF hub ID)",
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "fast1024", "full", "paper", "none"],
        default="fast",
        help="Inference preset. TODO: write detailed explanation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference, e.g. 'cuda' or 'cpu'. Auto-select if not set.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp16", "fp32"],
        default="fp16",
        help="Floating-point precision for inference",
    )
    parser.add_argument(
        "--color-maps",
        type=str,
        nargs="+",
        default=["Greys_r"],
        help="Color maps for visualization",
    )

    args = parser.parse_args()

    original_path = args.input_video
    rescaled_path = f"{args.output_dir}/{original_path.stem}_rescaled.mp4"

    print(f"Rescaling video to: {rescaled_path}")
    rescale_video(
        input_path=str(original_path),
        output_path=str(rescaled_path),
        width=1280,
        height=720,
    )

    print("Depth estimation...")
    predictor = DepthPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        color_maps=args.color_maps,
    )
    # depth_npy = predictor.predict(
    #     input_video_path=str(rescaled_path),
    #     output_dir=args.output_dir,
    # )
    depth_npy = np.load("outputs/generated_video_rescaled_pred.npy")

    print("Prepare FP inputs...")
    prepare_fp_inputs(
        rescaled_video_path=str(rescaled_path),
        mask_path="media/mask.png",
        cam_intrinsics_path="media/cam_K.txt",
        gt_depth_path="media/depth.raw",
        depth_npy=depth_npy,
        output_dir=args.output_dir,
        window_size=20,
        width=1280,
        height=720,
    )

    print("Pose rollout prediction...")
    # pose_rollout_predictor = PoseRolloutPredictor(
    #     data_path=args.output_dir,
    #     mesh_file="media/mesh/mesh.obj",
    #     est_refine_iter=10,
    #     track_refine_iter=4,
    #     debug=0,
    #     debug_dir="outputs/fp_outputs",
    # )
    # pose_rollout_predictor.run()

    vis = PoseTrajectoryVisualizer(
        pose_dir=f"{args.output_dir}/fp_outputs/ob_in_cam",
        depth_raw_path="media/depth.raw",
        intrinsics_path="media/cam_K.txt",
        width=1280,
        height=720,
    )
    vis.plot()
    vis.save_html(f"{args.output_dir}/trajectory_visualization.html")
    vis.show()


if __name__ == "__main__":
    main()
