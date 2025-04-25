import shutil
from pathlib import Path

import cv2
import numpy as np


def find_center_of_mask(mask_path: str, window_size: int = 20) -> np.ndarray:
    """
    Returns an array of (row, col) pixel coordinates within a square window
    of side `window_size` centered on the mask's centroid.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 0)
    center_r, center_c = int(np.median(ys)), int(np.median(xs))

    half = window_size // 2
    h, w = mask.shape
    coords = []
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            r = center_r + dr
            c = center_c + dc
            if 0 <= r < h and 0 <= c < w:
                coords.append([r, c])
    return np.array(coords)


def find_scale_and_shift(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    pixel_coords: np.ndarray,
    mask_invalid: bool = False,
) -> tuple[float, float]:
    """
    Estimate alpha, beta s.t. alpha * depth_pred + beta ≈ depth_gt
    using only the pixels specified by pixel_coords (row,col).
    """
    # resize first frame of prediction to match ground‐truth resolution
    pred0 = cv2.resize(
        depth_pred[0].astype(np.float32),
        (depth_gt.shape[1], depth_gt.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    rows = pixel_coords[:, 0]
    cols = pixel_coords[:, 1]

    dp_vals = pred0[rows, cols]
    dg_vals = depth_gt[rows, cols].astype(np.float32)

    if mask_invalid:
        valid = (dp_vals > 0) & (dg_vals > 0)
        dp_vals = dp_vals[valid]
        dg_vals = dg_vals[valid]

    # solve [dp_vals, 1] * [alpha; beta] = dg_vals
    A = np.stack([dp_vals, np.ones_like(dp_vals)], axis=-1)
    coefs, *_ = np.linalg.lstsq(A, dg_vals, rcond=None)
    alpha, beta = float(coefs[0]), float(coefs[1])
    return alpha, beta


def process_video_depth(
    pred_npy: str, alpha, beta, output_folder: str, width: int = 1280, height: int = 720
):
    out = Path(output_folder)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(pred_npy):
        d = cv2.resize(
            frame.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
        )
        d = alpha * d + beta
        d[d < 0] = 0
        fn = out / f"{i:06d}.png"
        cv2.imwrite(str(fn), d.astype(np.uint16))


def rescale_video(
    input_path: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: float | None = None,
    fourcc: str | None = None,
) -> None:
    """
    Rescales a video to the given width and height using OpenCV.

    Args:
        input_path: Path to the source video file.
        output_path: Path to save the rescaled video.
        width: Target frame width in pixels.
        height: Target frame height in pixels.
        fps: Desired output frames per second. If None, uses input video's FPS or defaults to 30.
        fourcc: FourCC codec string for VideoWriter (e.g., 'mp4v', 'XVID'). If None, defaults to 'mp4v'.
    """
    input_path = str(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {input_path}")

    # Determine FPS
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    out_fps = fps if fps is not None else (input_fps if input_fps > 0 else 30)

    # Setup writer
    codec_str = fourcc if fourcc else "mp4v"
    fourcc_code = cv2.VideoWriter_fourcc(*codec_str)
    writer = cv2.VideoWriter(str(output_path), fourcc_code, out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open VideoWriter for file: {output_path}")

    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (width, height))
        writer.write(resized)

    cap.release()
    writer.release()


def extract_frames(input_path: str, output_dir: str, ext: str = "png") -> None:
    """
    Extracts each frame of a video into an image file sequence.

    Args:
        input_path: Path to the source video.
        output_dir: Directory to save extracted frames; will be created.
        prefix: Filename prefix for frames, e.g., 'frame'.
        ext: Image file extension, e.g., 'png' or 'jpg'.
    """
    input_path = str(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Unable to open video file for frame extraction: {input_path}"
        )
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = output_dir / f"{idx:06d}.{ext}"
        cv2.imwrite(str(fname), frame)
        idx += 1
    cap.release()


def copy_mask(
    mask_path: str, output_mask_dir: str, frame_index: int = 0, ext: str = "png"
) -> None:
    """
    Copies a single mask image to the masks directory with zero-padded filename.

    Args:
        mask_path: Path to the source mask image (e.g., 'media/mask.png').
        output_mask_dir: Directory to save the copied mask.
        frame_index: Index for naming the mask file; default 0 to match first frame.
        ext: File extension for the copied mask.
    """
    src = Path(mask_path)
    if not src.is_file():
        raise FileNotFoundError(f"Mask file not found: {src}")
    dst_dir = Path(output_mask_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{frame_index:06d}.{ext}"
    shutil.copy(src, dst)


def prepare_fp_inputs(
    rescaled_video_path: str,
    mask_path: str,
    cam_intrinsics_path: str,
    gt_depth_path: str,
    depth_npy: str,
    output_dir: str,
    window_size: int = 20,
    width: int = 1280,
    height: int = 720,
) -> tuple[float, float]:
    """
    Runs the full FP input preparation:
      - extract frames,
      - copy mask,
      - copy intrinsics,
      - compute alpha & beta,
      - process depth frames.

    Returns:
        (alpha, beta)
    """
    fp = Path(output_dir)
    rgb_dir = fp / "rgb"
    masks_dir = fp / "masks"
    depth_dir = fp / "depth"

    extract_frames(rescaled_video_path, str(rgb_dir))
    copy_mask(mask_path, str(masks_dir))
    # copy intrinsics
    shutil.copy(cam_intrinsics_path, str(fp / Path(cam_intrinsics_path).name))

    # load ground truth
    gt = np.fromfile(gt_depth_path, dtype=np.uint16).reshape((height, width))

    # compute scale & shift
    pixel_coords = find_center_of_mask(str(masks_dir / "000000.png"), window_size)
    alpha, beta = find_scale_and_shift(depth_npy, gt, pixel_coords, mask_invalid=True)

    # process full depth video
    process_video_depth(depth_npy, alpha, beta, str(depth_dir), width, height)

    return alpha, beta
