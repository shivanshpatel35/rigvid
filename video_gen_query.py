#!/usr/bin/env python3
"""
Kling AI Image-to-Video Task Runner.

Generates a JWT, submits an image-to-video task, polls for completion,
and downloads the resulting video.
"""

import base64
import logging
import os
import time
from typing import Any, Dict, Optional

import jwt
import requests

# ─── Constants ────────────────────────────────────────────────────────────────

API_URL = "https://api.klingai.com/v1/videos/image2video"
POLL_INTERVAL = 5  # seconds between status checks
TOKEN_TTL = 1_800  # 30 minutes

# ─── Logging Setup ────────────────────────────────────────────────────────────


def setup_logging() -> None:
    """Configure root logger format and level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ─── Utility Functions ───────────────────────────────────────────────────────


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """
    Retrieve an environment variable or use a default.

    Args:
        name: Name of the environment variable.
        default: Value to return if the variable is unset.

    Returns:
        The value of the environment variable, or `default` if provided.

    Raises:
        EnvironmentError: if neither the env‑var nor default is set.
    """
    val = os.getenv(name, default)
    if val is None:
        raise EnvironmentError(f"Required environment variable not set: {name}")
    return val


# ─── Core Functionality ──────────────────────────────────────────────────────


def generate_api_token(access_key: str, secret_key: str) -> str:
    """
    Create a JWT for the Kling AI API.

    Args:
        access_key: Issuer.
        secret_key: HMAC signing key.

    Returns:
        Encoded JWT string.
    """
    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + TOKEN_TTL,
        "nbf": now - 5,
    }
    headers = {"alg": "HS256", "typ": "JWT"}
    return jwt.encode(payload, secret_key, algorithm="HS256", headers=headers)


def encode_image_to_base64(path: str) -> str:
    """
    Read an image file and return its Base64 representation.

    Args:
        path: Local image file path.

    Returns:
        Base64‐encoded string.
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_video_task(
    session: requests.Session,
    token: str,
    image_b64: str,
    prompt: str,
    model: str,
    duration: int,
) -> str:
    """
    Submit the image‐to‐video task.

    Returns:
        The task ID on success.

    Raises:
        RuntimeError on API failure.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "image": image_b64,
        "prompt": prompt,
        "negative_prompt": "Fast motion",
        "mode": "pro",
        "cfg_scale": 0.9,
        "model_version": model,
        "duration": duration,
    }
    resp = session.post(API_URL, json=payload, headers=headers)
    data = resp.json()
    if resp.status_code == 200 and data.get("code") == 0:
        task_id = data["data"]["task_id"]
        logging.info("Created task %s", task_id)
        return task_id
    raise RuntimeError(f"Create task failed: {data.get('message', resp.text)}")


def query_video_task(
    session: requests.Session, token: str, task_id: str
) -> Dict[str, Any]:
    """
    Poll the API for task status.

    Returns:
        Parsed JSON response.

    Raises:
        RuntimeError on API failure.
    """
    url = f"{API_URL}/{task_id}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = session.get(url, headers=headers)
    data = resp.json()
    if resp.status_code == 200 and data.get("code") == 0:
        return data
    raise RuntimeError(f"Query task failed: {data.get('message', resp.text)}")


def download_video(url: str, out_path: str) -> None:
    """
    Stream‐download the completed video.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8_192):
                f.write(chunk)
    logging.info("Downloaded video to %s", out_path)


def main() -> None:
    setup_logging()
    access_key = get_env_var("KLING_API_ACCESS_KEY")
    secret_key = get_env_var("KLING_API_SECRET_KEY")

    # Inputs
    image_path = get_env_var("IMAGE_PATH", "media/start_img.png")
    prompt = get_env_var("VIDEO_PROMPT", "Pour water on the plant")

    model = os.getenv("KLING_MODEL_VERSION", "kling-v1-6")
    duration = int(os.getenv("VIDEO_DURATION", "5"))
    output_path = os.getenv("VIDEO_OUTPUT_PATH", "outputs/generated_video.mp4")

    session = requests.Session()
    token = generate_api_token(access_key, secret_key)
    image_b64 = encode_image_to_base64(image_path)
    task_id = create_video_task(session, token, image_b64, prompt, model, duration)

    while True:
        result = query_video_task(session, token, task_id)
        status = result["data"]["task_status"]
        if status == "succeed":
            download_video(
                result["data"]["task_result"]["videos"][0]["url"], output_path
            )
            break
        if status == "failed":
            msg = result["data"].get("task_status_msg", "Unknown error")
            logging.error("Task %s failed: %s", task_id, msg)
            break
        logging.info(
            "Task %s in progress, retrying in %d seconds…", task_id, POLL_INTERVAL
        )
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
