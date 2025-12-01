import os
import numpy as np
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------
# utilities
# ------------------------------------------------

def load_episode_pose(episode_dir):
    pose_path = os.path.join(episode_dir, "pose", "poses.npz")
    assert os.path.exists(pose_path), f"Missing {pose_path}"

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
        # img = cv2.resize(img, resize)
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
    poses = load_episode_pose(episode_dir)
    images = load_episode_images(episode_dir)

    T = poses.shape[0]
    assert images.shape[0] == T, "Pose/Image length mismatch"

    # ---- states ----
    states = poses[:-1]

    # ---- actions ----
    actions = np.zeros_like(states)

    # 前 7 维：差分
    actions[:, :7] = poses[1:, :7] - poses[:-1, :7]

    # 最后一维：夹爪宽度（绝对值）
    actions[:, 7] = poses[1:, 7]

    # ---- images ----
    images = images[:-1]

    return states, actions, images


# ------------------------------------------------
# main conversion
# ------------------------------------------------

def convert_episodes_to_npz(episodes_root, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "train.npz")

    episode_names = sorted([
        d for d in os.listdir(episodes_root)
        if d.startswith("episode_")
    ])

    all_states, all_actions, all_images, traj_lengths = [], [], [], []

    for ep_name in episode_names:
        ep_dir = os.path.join(episodes_root, ep_name)
        print(f"Processing {ep_name}")

        states, actions, images = process_episode(ep_dir)

        all_states.append(states)
        all_actions.append(actions)
        all_images.append(images)
        traj_lengths.append(len(states))

    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    images = np.concatenate(all_images, axis=0)
    traj_lengths = np.array(traj_lengths, dtype=np.int64)

    np.savez_compressed(
        save_path,
        states=states,
        actions=actions,
        images=images,
        traj_lengths=traj_lengths
    )

    print(f"\nSaved dataset to {save_path}")
    print(f"states:  {states.shape}")
    print(f"actions: {actions.shape}")
    print(f"images:  {images.shape}")
    print(f"episodes: {len(traj_lengths)}")


# ------------------------------------------------
# entry
# ------------------------------------------------

if __name__ == "__main__":
    episodes_root = os.path.join(
        "/home/ani/Downloads/isaac_sim_data",
        "episodes_subsampled"
    )

    save_dir = os.path.join("/home/ani/Downloads/isaac_sim_data", "data", "isaac_rgb_masked")
    convert_episodes_to_npz(episodes_root, save_dir)
