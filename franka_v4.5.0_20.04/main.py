import os
import numpy as np
from PIL import Image
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera
from isaacsim.core.utils.types import ArticulationActions
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.utils.stage import open_stage, get_current_stage # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, \
euler_angles_to_quats,quats_to_euler_angles # type: ignore

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

asset_path = ROOT_DIR + "/franka_manipulation.usd"
# print(f"{asset_path=}")
urdf_path = "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/" \
"motion_policy_configs/FR3/fr3.urdf"
robot_description_path = "/home/ani/isaacsim/exts/isaacsim.robot_motion.motion_generation/" \
"motion_policy_configs/FR3/rmpflow/fr3_robot_description.yaml"

# exit()

# Prim path
marker_prim_path = "/_40_large_marker"
camera_prim_path = "/fr3/fr3_hand_tcp/hand"

def initial_camera(camera_path,frequency,resolution):
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
    
    f_stop = 0            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    focus_distance = 0.4    # in meters, the distance from the camera to the object plane

    horizontal_aperture =  20.955                   # The aperture size in mm
    vertical_aperture =  15.2908
    focal_length = 18.14756         # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_focus_distance(focus_distance)                   # The focus distance in meters
    camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_vertical_aperture(vertical_aperture / 10.0)

    # camera.set_clipping_range(0.01, 1.0e7)
    camera.set_clipping_range(0.1, 3.0)


    return camera

def save_camera_data(camera_key,data_dict, output_dir="./output_data"):
    """
    Save RGB and Depth data to files.
    
    Args:
        data_dict (dict): Dictionary with "rgb" and "depth" data.
        output_dir (str): Directory to save the files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB image
    rgb_image = Image.fromarray(data_dict["rgb"])
    rgb_image.save(os.path.join(output_dir, f"{camera_key}_rgb_image.png"))
    print(f"RGB image saved to {os.path.join(output_dir, f'{camera_key}_rgb_image.png')}")

def arm_const_speed(
    art, target_arm, sim,
    step_size=0.05, eps=5e-3
):
    current = art.get_joint_positions().squeeze()

    # print(f"{target_arm.shape=}")
    # print(f"{current.shape=}")
    # print(f"{current[:7].shape=}")

    while True:
        diff = target_arm - current[:7]
        dist = np.linalg.norm(diff)
        if dist < eps:
            break

        step = diff / (dist + 1e-8) * min(step_size, dist)
        cmd = current.copy()
        cmd[:7] += step

        art.apply_action(ArticulationActions(joint_positions=cmd))
        sim.step(render=True)

        current = art.get_joint_positions().squeeze()

def set_gripper(art, width, sim, steps=100):
    """
    set_gripper 's Docstring
    
    max width is 0.08
    """
    cmd = art.get_joint_positions().squeeze()
    print(f"{cmd=}")
    cmd[7] = width / 2
    cmd[8] = width / 2
    print(f"{cmd=}")
    action = ArticulationActions(joint_positions=cmd)

    for _ in range(steps):
        art.apply_action(action)
        sim.step(render=True)

def main():
    # Stage
    print("Opening stage...")
    open_stage(usd_path=asset_path)
    stage = get_current_stage()

    simulation_context = SimulationContext()
    simulation_context.initialize_physics()

    # Robot
    art = Articulation("/fr3")
    art.initialize()
    art_world_pose = art.get_world_poses()
    # print(f"{art.get_world_poses()}")
    # while art._is_initialized is not True:
    # print(f"{art._is_initialized=}")
    # dof_ptr = art.get_dof_index("fr3_joint1")
    # print(f"{art.dof_names}")
    # print(f"{art.body_names=}")
    # print("No problem !")
    # print(f"{art.get_joint_positions()=}")
    # exit()
    # initial_joint_position = np.array([[ 0.26720773,  0.25101955, -0.15940914, 
    #                                    -1.71126407,  0.00911494, 1.93678644,  0.90436569, 0.04, 0.04]])
    initial_joint_position = np.array([-0.47200201, -0.53468038,  0.41885995, -2.64197119,  0.24759319,
        2.1317271,  0.54534657,  0.04,  0.04])
    
    # exit()
    for i in range(50):
        art.set_joint_positions(initial_joint_position)
        simulation_context.step(render=True)

    gripper_efforts = np.array([-10, -10])
    art.set_joint_efforts(gripper_efforts, joint_indices=np.array([7, 8]))


    # Camera
    camera = initial_camera(camera_prim_path,60,(1920,1080))

    # exit()    

    # IK
    ik = LulaKinematicsSolver(
        robot_description_path=robot_description_path,
        urdf_path=urdf_path,
    )
    
    print("IKSolver get_all_frame_names:",ik.get_all_frame_names())
    print(f"{art._is_initialized=}")
    # AKSolver = ArticulationKinematicsSolver(art,ik,"fr3_hand_tcp")
    # print("<<< Set AKSolver successfully! >>>")

    # Object
    marker = XFormPrim(marker_prim_path)
    marker_pose = marker.get_world_pose()
    marker_position = marker_pose[0]
    marker_quat = marker_pose[1]
    print(f"marker position: {marker_position}")
    print(f"marker quat: {marker_quat}")

    # Target
    target_position = marker_position - art_world_pose[0]
    target_position = target_position.reshape(-1)
    # print(f"{target_position.shape=}")
    target_quat = np.array([0.0, 1.0, 0.0, 0.0])
    # exit()

    # Start simulation
    simulation_context.play()

    # Use to calculate initial point I want to go
    # target_position = target_position + np.array([0,0,0.3])
    # print(f"{target_position=}")
    # print(f"{target_position.shape=}")
    # exit()

    # Compute Inverse Kinematics 
    target_joint_position = ik.compute_inverse_kinematics("fr3_hand_tcp",target_position, target_quat)[0]
    target_joint_position = np.append(target_joint_position, np.array([0.04, 0.04]))
    print(f"{target_joint_position=}")

    # Take action
    arm_const_speed(art, target_joint_position[:7], simulation_context)
    set_gripper(art, width=0.0, sim=simulation_context,steps=100)

    target_joint_position[:7] = initial_joint_position[:7]
    arm_const_speed(art, target_joint_position[:7], simulation_context)
    
    end_joint_position = art.get_joint_positions().squeeze()
    print(f"{end_joint_position=}")

    simulation_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()