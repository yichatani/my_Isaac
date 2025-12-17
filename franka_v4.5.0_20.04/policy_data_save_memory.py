import os
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------
# utilities
# ------------------------------------------------

def load_episode_eef_pose(episode_dir):
    pose_path = os.path.join(episode_dir, "ee_pose", "ee_poses.npz")
    assert os.path.exists(pose_path), f"Missing {pose_path}"

    print(f"{pose_path=}")
    data = np.load(pose_path)
    poses = data["poses"]   # (T, 8)
    return poses

def load_episode_obj2camera_pose(episode_dir):
    pose_path = os.path.join(episode_dir, "obj_pose", "obj_poses.npz")
    assert os.path.exists(pose_path), f"Missing {pose_path}"

    print(f"{pose_path=}")
    data = np.load(pose_path)
    poses = data["poses"]   # (T, 8)
    return poses


def load_episode_images(episode_dir, resize=(96, 96)):
    rgb_dir = os.path.join(episode_dir, "rgb_masked")
    assert os.path.exists(rgb_dir), f"Missing {rgb_dir}"

    files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
    images = []

    for f in files:
        img = cv2.imread(os.path.join(rgb_dir, f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    images = np.array(images, dtype=np.uint8)           # (T, H, W, 3)
    images = np.transpose(images, (0, 3, 1, 2))          # (T, 3, H, W)

    return images


def process_episode(episode_dir):
    """
    Returns:
        states  (T-1, 8)
        actions (T-1, 8)
        images  (T-1, 3, H, W)
    """
    eef_poses = load_episode_eef_pose(episode_dir)
    marker_poses = load_episode_obj2camera_pose(episode_dir)
    images = load_episode_images(episode_dir)

    T = eef_poses.shape[0]
    assert images.shape[0] == T, "Pose/Image length mismatch"

    # ---- states ----
    states = marker_poses[:-1]

    # ---- actions ----
    actions = np.zeros_like(eef_poses[:-1])
    actions[:, :7] = eef_poses[1:, :7] - eef_poses[:-1, :7]
    actions[:, 7] = eef_poses[1:, 7]

    # ---- images ----
    images = images[:-1]

    return states, actions, images


# ------------------------------------------------
# 第一遍扫描：计算总长度和图像尺寸
# ------------------------------------------------

def scan_episodes(episodes_root):
    """扫描所有 episode，返回 episode 列表和每个的长度"""
    episode_names = sorted([
        d for d in os.listdir(episodes_root)
        if d.startswith("episode_")
    ])

    traj_lengths = []
    image_shape = None
    
    for ep_name in episode_names:
        ep_dir = os.path.join(episodes_root, ep_name)
        
        # 只加载 pose 来计算长度
        eef_poses = load_episode_eef_pose(ep_dir)
        traj_len = eef_poses.shape[0] - 1
        traj_lengths.append(traj_len)
        
        # 获取图像尺寸（只读第一张）
        if image_shape is None:
            rgb_dir = os.path.join(ep_dir, "rgb_masked")
            files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
            if len(files) > 0:
                img = cv2.imread(os.path.join(rgb_dir, files[0]), cv2.IMREAD_COLOR)
                H, W = img.shape[:2]
                image_shape = (3, H, W)  # (C, H, W)
        
        del eef_poses  # 释放内存
    
    return episode_names, traj_lengths, image_shape


# ------------------------------------------------
# 第二遍：逐个写入到 memory-mapped arrays
# ------------------------------------------------

def convert_episodes_to_npz(episodes_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "raw.npz")
    
    print("Step 1: Scanning episodes...")
    episode_names, traj_lengths, image_shape = scan_episodes(episodes_root)
    
    total_len = sum(traj_lengths)
    print(f"\nFound {len(episode_names)} episodes")
    print(f"Total trajectory length: {total_len}")
    print(f"Image shape: {image_shape}")
    
    # 创建 memory-mapped arrays
    print("\nStep 2: Creating memory-mapped arrays...")
    states_mmap = np.lib.format.open_memmap(
        os.path.join(save_dir, 'states.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(total_len, 7)
    )
    
    actions_mmap = np.lib.format.open_memmap(
        os.path.join(save_dir, 'actions.npy'),
        mode='w+',
        dtype=np.float32,
        shape=(total_len, 8)
    )
    
    images_mmap = np.lib.format.open_memmap(
        os.path.join(save_dir, 'images.npy'),
        mode='w+',
        dtype=np.uint8,
        shape=(total_len,) + image_shape
    )
    
    # 逐个处理并写入
    print("\nStep 3: Processing and writing episodes...")
    idx = 0
    for ep_name in episode_names:
        ep_dir = os.path.join(episodes_root, ep_name)
        print(f"Processing {ep_name}")
        
        states, actions, images = process_episode(ep_dir)
        traj_len = len(states)
        
        # 写入到 mmap
        states_mmap[idx:idx+traj_len] = states
        actions_mmap[idx:idx+traj_len] = actions
        images_mmap[idx:idx+traj_len] = images
        
        idx += traj_len
        
        # 释放内存
        del states, actions, images
    
    # 刷新到磁盘
    states_mmap.flush()
    actions_mmap.flush()
    images_mmap.flush()
    
    # 最后保存为 npz（读取 mmap，但一次性压缩写入）
    print("\nStep 4: Saving to compressed npz...")
    np.savez_compressed(
        save_path,
        states=states_mmap,
        actions=actions_mmap,
        images=images_mmap,
        traj_lengths=np.array(traj_lengths, dtype=np.int64)
    )
    
    # 清理临时文件
    del states_mmap, actions_mmap, images_mmap
    os.remove(os.path.join(save_dir, 'states.npy'))
    os.remove(os.path.join(save_dir, 'actions.npy'))
    os.remove(os.path.join(save_dir, 'images.npy'))
    
    print(f"\n✓ Saved dataset to {save_path}")
    print(f"  states:  ({total_len}, 7)")
    print(f"  actions: ({total_len}, 8)")
    print(f"  images:  ({total_len}, {image_shape[0]}, {image_shape[1]}, {image_shape[2]})")
    print(f"  episodes: {len(traj_lengths)}")


# ------------------------------------------------
# entry
# ------------------------------------------------

if __name__ == "__main__":
    episodes_root = "/portal/test_data/data_12_16_aug"
    save_dir = os.path.join(episodes_root, "data_seg")
    convert_episodes_to_npz(episodes_root, save_dir)