import os
import time
import zmq
import cv2
import numpy as np
from collections import deque
from isaacsim import SimulationApp

# =====================================================================
# Simulation App
# =====================================================================
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from omni.isaac.sensor import Camera
from isaacsim.core.utils.types import ArticulationActions
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, quats_to_rot_matrices

# =====================================================================
# Config
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
asset_path = ROOT_DIR + "/franka_manipulation.usd"

urdf_path = (
    "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/FR3/fr3.urdf"
)
robot_description_path = (
    "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/"
    "motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml"
)

camera_prim_path = "/fr3/fr3_hand_tcp/hand"

ZMQ_ADDR = "tcp://127.0.0.1:5556"

TARGET_H, TARGET_W = 448, 448

OBS_STEPS = 2
IMG_STEPS = 2
ACTION_HORIZON = 4

# =====================================================================
# Policy Client
# =====================================================================
class PolicyClient:
    def __init__(self, addr):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.connect(addr)
        print(f"[PolicyClient] connected to {addr}")

    def infer(self, obs):
        self.sock.send_pyobj(obs)
        return self.sock.recv_pyobj()  # (H, 8)

# =====================================================================
# Camera
# =====================================================================
def initial_camera(camera_path, frequency, resolution):
    """Initialize Camera"""
    camera = Camera(
        prim_path=camera_path,
        frequency=frequency,
        resolution=resolution,
    )

    camera.initialize()
    camera.add_motion_vectors_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera = set_camera_parameters(camera)

    print("Camera Initialized!")

    return camera

def set_camera_parameters(camera):
    
    f_stop = 0
    focus_distance = 0.4

    horizontal_aperture = 20.955
    vertical_aperture = 15.2908
    focal_length = 18.14756

    camera.set_focal_length(focal_length / 10.0)
    camera.set_focus_distance(focus_distance)
    camera.set_lens_aperture(f_stop * 100.0)
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)
    camera.set_vertical_aperture(vertical_aperture / 10.0)
    camera.set_clipping_range(0.1, 3.0)

    return camera

def get_rgb(cam):
    frame = cam.get_current_frame()
    if frame is None:
        return None
    rgba = frame.get("rgba", None)
    if not isinstance(rgba, np.ndarray):
        return None
    if rgba.ndim != 3 or rgba.shape[2] < 3:
        return None
    if rgba.shape[0] == 0 or rgba.shape[1] == 0:
        return None
    rgb = rgba[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8, copy=False)
    try:
        rgb = cv2.resize(
            rgb,
            (TARGET_W, TARGET_H),
            interpolation=cv2.INTER_LINEAR
        )
    except cv2.error:
        return None
    return rgb


# =====================================================================
# Robot State
# =====================================================================
def get_tcp_pose(art, ik):
    q = art.get_joint_positions().squeeze()
    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]
    width = q[7] + q[8]
    return np.concatenate([np.array(pos), quat, [width]]).astype(np.float32)

# =====================================================================
# Apply delta-pose action (LOCAL TCP FRAME)
# =====================================================================
def apply_delta_action(art, ik, delta):
    q = art.get_joint_positions().squeeze()

    # -----------------------------
    # arm: delta pose
    # -----------------------------
    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    pos = np.array(pos)
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]

    dp = delta[:3]
    dquat = delta[3:7]
    gripper_width = float(delta[7])   # ✅ absolute

    target_pos = pos + dp
    target_quat = quat + dquat
    target_quat /= np.linalg.norm(target_quat + 1e-8)

    q_target = ik.compute_inverse_kinematics(
        "fr3_hand_tcp", target_pos, target_quat
    )[0]

    # -----------------------------
    # build full joint command
    # -----------------------------
    cmd = q.copy()
    cmd[:7] = q_target

    gripper_width = np.clip(gripper_width, 0.0, 0.08)  # safety
    cmd[7] = gripper_width / 2.0
    cmd[8] = gripper_width / 2.0

    art.apply_action(ArticulationActions(joint_positions=cmd))


# =====================================================================
# Main Loop
# =====================================================================
def main():
    open_stage(asset_path)

    sim = SimulationContext()
    sim.initialize_physics()

    art = Articulation("/fr3")
    art.initialize()
    initial_joint_position = np.array([-0.47200201, -0.53468038, 0.41885995, -2.64197119, 0.24759319,
                                       2.1317271, 0.54534657, 0.04, 0.04])
    for i in range(50):
        art.set_joint_positions(initial_joint_position)
        sim.step(render=True)

    ik = LulaKinematicsSolver(
        urdf_path=urdf_path,
        robot_description_path=robot_description_path,
    )

    camera = initial_camera(camera_prim_path, 60, (1920, 1080))

    policy = PolicyClient(ZMQ_ADDR)

    state_buf = deque(maxlen=OBS_STEPS)
    img_buf = deque(maxlen=IMG_STEPS)

    sim.play()

    print("[Isaac] Policy-driven control started")

    while sim.is_playing():

        rgb = get_rgb(camera)
        pose = get_tcp_pose(art, ik)

        if rgb is None:
            sim.step(render=True)
            continue

        state_buf.append(pose)
        img_buf.append(rgb)

        if len(state_buf) < OBS_STEPS or len(img_buf) < IMG_STEPS:
            sim.step(render=True)
            continue

        obs = {
            "state": np.stack(state_buf, axis=0),   # (2,8)
            "image": np.stack(img_buf, axis=0),     # (2,H,W,3)
        }

        action_seq = policy.infer(obs)  # (H,8)
        delta = action_seq[0]

        apply_delta_action(art, ik, delta)
        sim.step(render=True)

    sim.stop()
    simulation_app.close()

# =====================================================================
if __name__ == "__main__":
    main()
