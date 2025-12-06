import os
import cv2
import numpy as np
from tqdm import tqdm

def apply_mask_to_rgb(
    episode_dir,
    rgb_dir="rgb",
    mask_dir="mask",
    output_dir="rgb_masked",
):
    rgb_path = os.path.join(episode_dir, rgb_dir)
    mask_path = os.path.join(episode_dir, mask_dir)
    out_path = os.path.join(episode_dir, output_dir)

    assert os.path.exists(rgb_path), f"{rgb_path} not found"
    assert os.path.exists(mask_path), f"{mask_path} not found"

    os.makedirs(out_path, exist_ok=True)

    rgb_files = sorted([
        f for f in os.listdir(rgb_path)
        if f.endswith(".png")
    ])

    print(f"Processing {len(rgb_files)} frames...")

    for fname in tqdm(rgb_files):
        rgb_file = os.path.join(rgb_path, fname)
        mask_file = os.path.join(mask_path, fname.replace("rgb", "mask"))

        if not os.path.exists(mask_file):
            print(f"Warning: mask not found for {fname}, skip")
            continue

        # ---- load ----
        rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)  # BGR
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        if rgb is None or mask is None:
            print(f"Warning: failed to load {fname}")
            continue

        # ---- ensure mask shape ----
        if mask.ndim == 3:
            mask = mask[..., 0]

        # ---- binarize mask ----
        # 支持 0/255 或 0/1
        mask = (mask > 0).astype(np.uint8)

        # ---- apply mask ----
        mask_3c = mask[:, :, None]   # (H, W, 1)
        rgb_masked = rgb * mask_3c   # broadcast

        # ---- save ----
        out_file = os.path.join(out_path, fname)
        cv2.imwrite(out_file, rgb_masked)

    print(f"Masked RGB saved to: {out_path}")


if __name__ == "__main__":
    episode_dir = "/home/ani/Downloads/episodes_subsampled_2/episode_0001"
    apply_mask_to_rgb(episode_dir)
