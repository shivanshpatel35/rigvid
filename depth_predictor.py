"""
Depth predictor using RollingDepthPipeline to infer and save depth maps and visualization videos.
"""

import sys

sys.path.insert(0, "RollingDepth")
import logging
import os
from pathlib import Path

import einops
import numpy as np
import torch
from rollingdepth import (RollingDepthPipeline,
                          concatenate_videos_horizontally_torch, get_video_fps,
                          write_video_from_numpy)
from src.util.colorize import colorize_depth_multi_thread


class DepthPredictor:
    def __init__(
        self,
        checkpoint: str = "prs-eth/rollingdepth-v1-0",
        device: str | None = None,
        dtype: str = "fp16",
        color_maps: list[str] | None = None,
        save_sbs: bool = False,
    ):
        """
        Args:
            checkpoint: model checkpoint path or HF hub ID
            device: 'cpu' or 'cuda'; if None, auto-select
            dtype: 'fp16' or 'fp32'
            color_maps: list of matplotlib colormap names for visualization
            save_sbs: whether to save side-by-side RGB-depth videos
        """
        # Logger
        logging.basicConfig(level=logging.INFO)

        # Device setup
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load pipeline
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        self.pipe = RollingDepthPipeline.from_pretrained(
            checkpoint, torch_dtype=torch_dtype
        )
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except ImportError:
            logging.warning("xformers not available, running without it")
        self.pipe = self.pipe.to(self.device)

        # Visualization settings
        self.color_maps = color_maps or ["Spectral_r", "Greys_r"]
        self.save_sbs = save_sbs
        self.depth_predictor_kwargs = {
            "dilations": [1, 25],
            "refine_step": 0,
        }

    def predict(
        self,
        input_video_path: str | Path,
        output_dir: str | Path,
    ):
        """
        Run depth prediction on a single video and save outputs.

        Args:
            input_video_path: path to the input video file
            output_dir: directory to save .npy and video outputs
            pipeline_kwargs: keyword args passed to the pipeline call (e.g., dilations, snippet_lengths)
        """
        input_path = Path(input_video_path)
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Inference
        logging.info(f"Predicting depth for {input_path}")
        with torch.no_grad():
            pipe_out = self.pipe(
                input_video_path=str(input_path),
                **self.depth_predictor_kwargs,
            )

        # Depth tensor [N,1,H,W]
        depth_pred = pipe_out.depth_pred.cpu()

        # Save raw depth maps
        npy_out = output_dir / f"{input_path.stem}_pred.npy"
        np.save(npy_out, depth_pred.numpy().squeeze(1))  # [N,H,W]

        # Colorize and save videos
        fps = int(get_video_fps(str(input_path)))
        for cmap in self.color_maps:
            if not cmap:
                continue
            colored = colorize_depth_multi_thread(
                depth=depth_pred.numpy(),
                valid_mask=None,
                chunk_size=4,
                num_threads=4,
                color_map=cmap,
                verbose=False,
            )  # [N,H,W,3]

            # Save colored depth video
            vid_out = output_dir / f"{input_path.stem}_{cmap}.mp4"
            write_video_from_numpy(
                frames=colored,
                output_path=str(vid_out),
                fps=fps,
                crf=23,
                preset="medium",
                verbose=False,
            )

            # Save side-by-side RGB-depth video
            if self.save_sbs and cmap == self.color_maps[0]:
                rgb = pipe_out.input_rgb.cpu() * 255  # [N,3,H,W]
                colored_depth = einops.rearrange(
                    torch.from_numpy(colored), "n h w c -> n c h w"
                )
                concat = (
                    concatenate_videos_horizontally_torch(
                        rgb.int(), colored_depth.int(), gap=10
                    )
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )  # [N,3,H,W]
                concat = einops.rearrange(concat, "n c h w -> n h w c")
                rgbd_out = output_dir / f"{input_path.stem}_rgbd.mp4"
                write_video_from_numpy(
                    frames=concat,
                    output_path=str(rgbd_out),
                    fps=fps,
                    crf=23,
                    preset="medium",
                    verbose=False,
                )

        return depth_pred.numpy().squeeze(1)
