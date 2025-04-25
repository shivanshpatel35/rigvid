# visualizer.py
"""
Module for visualizing 6-DoF pose rollout trajectories and colored scene point cloud in 3D.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.spatial.transform import Rotation as R


def load_intrinsics(file_path: str) -> dict:
    """
    Load camera intrinsic parameters from a text file (3x3 or fx fy cx cy format).
    Returns dict with fx, fy, cx, cy.
    """
    data = np.loadtxt(file_path)
    if data.shape == (3, 3):
        fx = data[0, 0]
        fy = data[1, 1]
        cx = data[0, 2]
        cy = data[1, 2]
    elif data.size == 4:
        fx, fy, cx, cy = data
    else:
        raise ValueError("Unsupported intrinsics format")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def depth_to_point_cloud(
    depth_raw_path: str, intrinsics: dict, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert raw depth file to point cloud and pixel colors in camera frame.

    Returns:
        pts: Nx3 array of 3D points (m)
        colors: Nx3 array of RGB colors (0-255)
    """
    # load depth
    depth = (
        np.fromfile(depth_raw_path, dtype=np.uint16).reshape((height, width)) / 1000.0
    )
    # load image for color
    # assume same resolution as depth
    img = cv2.imread("media/start_img.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fx, fy, cx, cy = (
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["cx"],
        intrinsics["cy"],
    )

    # grid of pixel coords
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy

    pts = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = img_rgb.reshape(-1, 3)

    # mask valid
    valid = (pts[:, 2] > 0) & (pts[:, 2] < 1) & (pts[:, 0] > -0.45) & (pts[:, 0] < 0.45)

    pts = pts[valid]
    colors = colors[valid]

    return pts, colors


class PoseTrajectoryVisualizer:
    """
    Loads poses and scene point cloud, then visualizes both in 3D.
    """

    def __init__(
        self,
        pose_dir: str,
        depth_raw_path: str,
        intrinsics_path: str,
        width: int,
        height: int,
        subsample_pcd: int = 100000,
    ):
        self.pose_dir = Path(pose_dir)
        self.depth_raw_path = depth_raw_path
        self.intrinsics_path = intrinsics_path
        self.width = width
        self.height = height
        self.subsample_pcd = subsample_pcd
        self.positions = None
        self.fig = go.Figure()

    def load_poses(self):
        poses = []
        for file in sorted(self.pose_dir.glob("*.txt")):
            T = np.loadtxt(file)
            t = T[:3, 3]
            poses.append(t)
        self.positions = np.stack(poses, axis=0)

    def plot(self):
        # point cloud
        intr = load_intrinsics(self.intrinsics_path)
        pts, colors = depth_to_point_cloud(
            self.depth_raw_path, intr, self.width, self.height
        )
        if pts.shape[0] > self.subsample_pcd:
            idx = random.sample(range(pts.shape[0]), self.subsample_pcd)
            pts = pts[idx]
            colors = colors[idx]
        color_strs = [f"rgb({r},{g},{b})" for r, g, b in colors]
        self.fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=1, color=color_strs),
                name="Scene PCD",
            )
        )
        # trajectory
        if self.positions is None:
            self.load_poses()
        pos = self.positions
        self.fig.add_trace(
            go.Scatter3d(
                x=pos[:, 0],
                y=pos[:, 1],
                z=pos[:, 2],
                mode="markers+lines",
                marker=dict(size=4, color="red"),
                line=dict(width=2, color="red"),
                name="Object Trajectory",
            )
        )
        self.fig.update_layout(
            scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
            title="Scene PCD + Object Trajectory",
            showlegend=True,
        )

    def show(self):
        self.fig.show()

    def save_html(self, filepath: str):
        pio.write_html(self.fig, file=filepath)
