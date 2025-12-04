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
simulation_app = SimulationApp({
    "headless": False,
    "renderer": "Storm"
})

import carb.settings
settings = carb.settings.get_settings()
settings.set_bool("/rtx/enabled", False)
settings.set_bool("/rtx/reflections/enabled", False)
settings.set_bool("/rtx/translucency/enabled", False)
settings.set_bool("/rtx/postProcessing/enabled", False)
settings.set_bool("/rtx/dlss/enabled", False)
settings.set_bool("/rtx/caustics/enabled", False)
settings.set_bool("/rtx/ambientOcclusion/enabled", False)

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from omni.isaac.sensor import Camera
from isaacsim.core.utils.types import ArticulationActions
from omni.isaac.motion_generation import LulaKinematicsSolver
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats

# =====================================================================
# Config
# =====================================================================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
asset_path = ROOT_DIR + "/franka_manipulation.usd"

# SEG_ADDR = "tcp://192.168.56.55:5556"     # seg server
SEG_ADDR = "tcp://192.168.56.56:5555"
ACTION_BIND = "tcp://192.168.56.56:5557"  # policy → isaac

TARGET_H, TARGET_W = 448, 448
OBS_STEPS = 2
IMG_STEPS = 2

# =====================================================================
# ZMQ Setup (DUAL SOCKET)
# =====================================================================
ctx = zmq.Context.instance()

# → SEG (send obs)
seg_sock = ctx.socket(zmq.REQ)
seg_sock.connect(SEG_ADDR)
print(f"[Isaac] Connected to SegServer {SEG_ADDR}")

# ← POLICY (receive action)
action_sock = ctx.socket(zmq.REP)
action_sock.bind(ACTION_BIND)
print(f"[Isaac] Waiting for Policy on {ACTION_BIND}")

# =====================================================================
# Camera utilities
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
    if rgba is None:
        return None
    rgb = rgba[:, :, :3].astype(np.uint8)
    return cv2.resize(rgb, (TARGET_W, TARGET_H))

# =====================================================================
# Robot helpers
# =====================================================================
def get_tcp_pose(art, ik):
    q = art.get_joint_positions().squeeze()
    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]
    width = q[7] + q[8]
    return np.concatenate([pos, quat, [width]]).astype(np.float32)

def apply_delta_action(art, ik, delta):
    q = art.get_joint_positions().squeeze()
    pos, rot = ik.compute_forward_kinematics("fr3_hand_tcp", q[:7])
    quat = rot_matrices_to_quats(np.array(rot).reshape(1, 3, 3))[0]

    target_pos = pos + delta[:3]
    target_quat = quat + delta[3:7]
    target_quat /= np.linalg.norm(target_quat + 1e-8)

    q_target = ik.compute_inverse_kinematics(
        "fr3_hand_tcp", target_pos, target_quat
    )[0]

    cmd = q.copy()
    cmd[:7] = q_target
    cmd[7] = cmd[8] = np.clip(delta[7], 0.0, 0.08) / 2.0

    for _ in range(20):
        art.apply_action(ArticulationActions(joint_positions=cmd))

# =====================================================================
# Main
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
        urdf_path="/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3/fr3.urdf",
        robot_description_path="/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml",
    )

    camera = initial_camera("/fr3/fr3_hand_tcp/hand", 15, (1920, 1080))


    state_buf = deque(maxlen=OBS_STEPS)
    img_buf = deque(maxlen=IMG_STEPS)
    action_queue = deque()

    sim.play()
    print("[Isaac] Dual-socket control started")

    for _ in range(50):
        sim.step(render=True)

    while sim.is_playing():
        rgb = get_rgb(camera)
        pose = get_tcp_pose(art, ik)

        if rgb is None or pose is None:
            sim.step(render=True)
            continue

        state_buf.append(pose)
        img_buf.append(rgb)

        if len(action_queue) == 0:
            if len(state_buf) >= OBS_STEPS and len(img_buf) >= IMG_STEPS:
                obs = {
                    "state": np.stack(state_buf, axis=0),
                    "image": np.stack(img_buf, axis=0),
                }

                seg_sock.send_pyobj(obs)
                seg_sock.recv()

                action_seq = action_sock.recv_pyobj()
                action_sock.send(b"ok")
                
                # print(f"Received {len(action_seq)} actions")
                print(f"Received action: {action_seq[0]}")
                action_queue.extend(action_seq)
            else:
                if len(state_buf) < OBS_STEPS or len(img_buf) < IMG_STEPS:
                    print("Waiting for observation buffer to fill...")

        if len(action_queue) > 0:
            delta = action_queue.popleft()
            apply_delta_action(art, ik, delta)

        sim.step(render=True)

    sim.stop()
    simulation_app.close()

if __name__ == "__main__":
    main()


# sent_once = False
# waiting_for_action = False
# while sim.is_playing():
#         rgb = get_rgb(camera)
#         pose = get_tcp_pose(art, ik)

#         # -----------------------
#         # 1. obs validity gate
#         # -----------------------
#         if rgb is None or pose is None:
#             sim.step(render=True)
#             continue

#         state_buf.append(pose)
#         img_buf.append(rgb)

#         if len(state_buf) < OBS_STEPS or len(img_buf) < IMG_STEPS:
#             sim.step(render=True)
#             continue

#         obs = {
#             "state": np.stack(state_buf, axis=0),
#             "image": np.stack(img_buf, axis=0),
#         }

#         # -----------------------
#         # 2. send obs → seg
#         # -----------------------
#         seg_sock.send_pyobj(obs)
#         seg_sock.recv()

#         sent_once = True
#         waiting_for_action = True

#         # -----------------------
#         # 3. wait action ONLY if sent_once
#         # -----------------------
#         if waiting_for_action:
#             action_seq = action_sock.recv_pyobj()
#             action_sock.send(b"ok")

#             delta = action_seq[0]

#             print(f"{delta.shape=}")
#             print(f"{delta=}")

#             # for delta in action_seq:
#             #     apply_delta_action(art, ik, delta)
#             apply_delta_action(art, ik, delta)
#             waiting_for_action = False

#         sim.step(render=True)
