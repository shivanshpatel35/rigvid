# 6d_pose_rollout_predictor.py
"""
Script for 6-DoF pose rollout using FoundationPose: run registration on first frame and tracking on subsequent frames,
visualize and save results, then compile tracking video.
"""
import glob
import logging
import os
import shutil
import sys

import cv2
import imageio
import numpy as np
import trimesh

sys.path.insert(0, "FoundationPose")
from datareader import *
from estimater import *


def create_video_from_images(image_dir, output_video_path, fps=30):
    # Get list of image files in sorted order
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

    if not image_files:
        print("No images found in the directory.")
        return

    # Read the first image to get video dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' for .mp4 output
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for img_path in image_files:
        image = cv2.imread(img_path)
        video_writer.write(image)

    # Release the video writer
    video_writer.release()
    print(f"Video saved at {output_video_path}")


class PoseRolloutPredictor:
    """
    Encapsulates the registration and tracking of 6-DoF object pose over a video sequence.
    """

    def __init__(
        self,
        data_path: str,
        mesh_file: str,
        est_refine_iter: int = 10,
        track_refine_iter: int = 4,
        debug: int = 2,
        debug_dir: str = "debug",
    ):
        # store parameters
        self.data_path = data_path
        self.mesh_file = mesh_file
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.debug_dir = debug_dir

        # init logging and seed
        set_logging_format()
        set_seed(0)

        # prepare debug dirs
        if os.path.isdir(self.debug_dir):
            shutil.rmtree(self.debug_dir)
        os.makedirs(os.path.join(self.debug_dir, "track_vis"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "ob_in_cam"), exist_ok=True)

        # load mesh and bbox
        mesh = trimesh.load(self.mesh_file)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.mesh = mesh
        self.to_origin = to_origin
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # instantiate estimator
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx,
        )
        logging.info("FoundationPose estimator ready.")

        # set up reader
        self.reader = YcbineoatReader(
            video_dir=self.data_path, shorter_side=None, zfar=np.inf
        )

    def run(self):
        """
        Runs registration on the first frame and tracking on subsequent frames.
        Saves poses, visualizations, and compiles tracking video.
        """
        num_frames = len(self.reader.color_files)
        logging.info(f"Running on {num_frames} frames")
        for i in range(num_frames):
            color = self.reader.get_color(i)
            depth = self.reader.get_depth(i)
            logging.info(f"Frame {i}")
            if i == 0:
                mask = self.reader.get_mask(0).astype(bool)
                pose = self.estimator.register(
                    K=self.reader.K,
                    rgb=color,
                    depth=depth,
                    ob_mask=mask,
                    iteration=self.est_refine_iter,
                )
                if self.debug >= 3:
                    # export model transform and point cloud
                    m = self.mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{self.debug_dir}/model_tf.obj")
                    xyz_map = depth2xyzmap(depth, self.reader.K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                    pcd_path = os.path.join(self.debug_dir, "scene_complete.ply")
                    import open3d as o3d

                    o3d.io.write_point_cloud(pcd_path, pcd)
            else:
                pose = self.estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=self.reader.K,
                    iteration=self.track_refine_iter,
                )
            # save pose matrix
            pose_path = os.path.join(
                self.debug_dir, "ob_in_cam", f"{self.reader.id_strs[i]}.txt"
            )
            np.savetxt(pose_path, pose.reshape(4, 4))

            # visualization
            if self.debug >= 1:
                center_pose = pose @ np.linalg.inv(self.to_origin)
                vis = draw_posed_3d_box(
                    self.reader.K,
                    img=color,
                    ob_in_cam=center_pose,
                    bbox=self.bbox,
                    line_color=(0, 255, 0),
                    linewidth=3,
                )
                vis = draw_xyz_axis(
                    color=vis,
                    ob_in_cam=center_pose,
                    scale=0.1,
                    K=self.reader.K,
                    thickness=3,
                    transparency=0,
                )
                # save and/or show
                if self.debug >= 2:
                    out_path = os.path.join(
                        self.debug_dir, "track_vis", f"{self.reader.id_strs[i]}.png"
                    )
                    imageio.imwrite(out_path, vis)
                if self.debug >= 1:
                    cv2.imshow('Track', vis[..., ::-1]); cv2.waitKey(1)

        # compile video
        if self.debug >= 2:
            create_video_from_images(
                os.path.join(self.debug_dir, "track_vis"),
                os.path.join(self.debug_dir, "tracking_video.mp4"),
                fps=30,
            )
            shutil.rmtree(os.path.join(self.debug_dir, "track_vis"))
