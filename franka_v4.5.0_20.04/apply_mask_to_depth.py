import os
import cv2
import numpy as np
from tqdm import tqdm

def apply_mask_to_depth(
    episode_dir,
    depth_dir="depth",
    mask_dir="mask",
    output_dir="depth_masked",
):
    depth_path = os.path.join(episode_dir, depth_dir)
    mask_path = os.path.join(episode_dir, mask_dir)
    out_path = os.path.join(episode_dir, output_dir)
    
    assert os.path.exists(depth_path), f"{depth_path} not found"
    assert os.path.exists(mask_path), f"{mask_path} not found"
    os.makedirs(out_path, exist_ok=True)
    
    depth_files = sorted([
        f for f in os.listdir(depth_path)
        if f.endswith(".png")
    ])
    
    print(f"Processing {len(depth_files)} depth frames...")
    
    for fname in tqdm(depth_files):
        depth_file = os.path.join(depth_path, fname)
        mask_file = os.path.join(mask_path, fname.replace("depth", "mask"))
        
        if not os.path.exists(mask_file):
            print(f"Warning: mask not found for {fname}, skip")
            continue
        
        # ---- load ----
        # Use IMREAD_UNCHANGED to preserve depth encoding (uint16)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        
        if depth is None or mask is None:
            print(f"Warning: failed to load {fname}")
            continue
        
        # ---- ensure mask shape ----
        if mask.ndim == 3:
            mask = mask[..., 0]
        
        # ---- binarize mask ----
        mask = (mask > 0).astype(np.uint8)
        
        # ---- apply mask ----
        # For depth, we want to set masked-out pixels to 0 (invalid depth)
        depth_masked = depth * mask
        
        # ---- save ----
        out_file = os.path.join(out_path, fname)
        cv2.imwrite(out_file, depth_masked)
    
    print(f"Masked depth saved to: {out_path}")


if __name__ == "__main__":
    for i in range(100):
        episode_dir = f"/portal/test_data/episodes_12_17/episodes_sync_B_mode_aug/episode_0{i:03d}"
        apply_mask_to_depth(episode_dir)