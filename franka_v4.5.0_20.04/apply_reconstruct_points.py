import os
import numpy as np
from tqdm import tqdm
from process_data import reconstruct_pointcloud


def save_all_pointclouds(
    episode_dir,
    output_dir="pointclouds",
    start_frame=0,
    end_frame=None,
):
    """
    重建episode中所有帧的点云并保存
    
    Args:
        episode_dir: episode目录路径
        output_dir: 点云保存的子目录名
        start_frame: 起始帧索引
        end_frame: 结束帧索引（None表示处理所有帧）
    """
    rgb_path = os.path.join(episode_dir, "rgb_masked")
    out_path = os.path.join(episode_dir, output_dir)
    
    assert os.path.exists(rgb_path), f"{rgb_path} not found"
    os.makedirs(out_path, exist_ok=True)
    
    rgb_files = sorted([
        f for f in os.listdir(rgb_path)
        if f.endswith(".png")
    ])
    
    if end_frame is None:
        end_frame = len(rgb_files)
    
    frame_indices = range(start_frame, min(end_frame, len(rgb_files)))
    
    print(f"Processing {len(frame_indices)} frames in {os.path.basename(episode_dir)}...")
    
    for frame_idx in tqdm(frame_indices):
        try:
            # reconstruct point cloud
            point_cloud = reconstruct_pointcloud(
                episode_dir,
                frame_idx=frame_idx,
                visualize=False
            )
            
            # save as .npy format
            out_file = os.path.join(out_path, f"pointcloud_{frame_idx:06d}.npy")
            np.save(out_file, point_cloud)
            
        except Exception as e:
            print(f"Warning: failed to process frame {frame_idx}: {e}")
            continue
    
    print(f"Point clouds saved to: {out_path}")


if __name__ == "__main__":
    # process single
    # episode_dir = "/portal/test_data/episodes_12_17/episodes_sync_B_mode_aug/episode_0000"
    # save_all_pointclouds(episode_dir)
    
    base_dir = "/portal/test_data/episodes_12_17/episodes_sync_B_mode_aug"
    
    for i in range(100):
        episode_dir = os.path.join(base_dir, f"episode_{i:04d}")
        
        if not os.path.exists(episode_dir):
            print(f"Skipping {episode_dir} (not found)")
            continue
            
        save_all_pointclouds(episode_dir)
    
    print("All episodes processed!")