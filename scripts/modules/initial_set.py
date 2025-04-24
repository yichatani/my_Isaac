import os
import time
import omni.usd # type: ignore
from PIL import Image
import math
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from omni.isaac.core.prims import XFormPrim # type: ignore
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore
from omni.isaac.core.simulation_context import SimulationContext # type: ignore
from omni.isaac.sensor import Camera # type: ignore
from omni.isaac.core import World # type: ignore


# Function to list all prims in the stage
def get_all_prim_paths(stage):
    stage = omni.usd.get_context().get_stage()
    prim_paths = []
    for prim in stage.Traverse():
        prim_paths.append(prim.GetPath().pathString)  # Get prim path as string
    for path in prim_paths:
        print(path)

def find_joint_prim_path_by_name(dof_name: str, root_path="/World"):
    """
    Find the joint prim path by its name in the USD stage.
    Args:
        dof_name (str): The name of the joint to search for.
        root_path (str): The root path to start the search from.
    Returns:
        str: The prim path of the joint if found, otherwise None.
    """
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == dof_name and prim.GetTypeName() == "PhysicsRevoluteJoint":
            return prim.GetPath().pathString
    return None


def reset_obj_pose(prim_paths,simulation_context):
    for prim_path in prim_paths:
        obj = XFormPrim(prim_path)
        euler_angles = [
            0,
            0,
            # random.uniform(-math.pi/2, math.pi/2),
            # random.uniform(-math.pi/2, math.pi/2),
            random.uniform(-math.pi, math.pi),
            # 0
        ]
        obj.set_world_pose(
            position=[
                # random.uniform(0.33, 1.10), 
                # random.uniform(-0.15,0.55), 
                # random.uniform(0.8,0.85)
                random.uniform(0.6,0.9),
                random.uniform(-0.11,0.4),
                random.uniform(0.90,1.00)             
            ],
            orientation = tuple(R.from_euler('xyz', euler_angles).as_quat())
        )
        for _ in range(20):
            simulation_context.step(render=True)
    for _ in range(100):
        simulation_context.step(render=True)
    print("Reset the objects' positions!")


def check_obj_pose_err(prim_paths:list) -> None:
    for prim_path in prim_paths:
        obj = XFormPrim(prim_path)
        if obj.get_world_pose()[0][2] < 1.00 and \
            0.6 < obj.get_world_pose()[0][0] < 0.9 and \
            -0.11 < obj.get_world_pose()[0][1] < 0.4: 
            return False
    print("All objects are in the wrong position!")
    return True

def reset_obj_z(prim_paths:list,simulation_context) -> None:
    for prim_path in prim_paths:
        obj = XFormPrim(prim_path)
        if obj.get_world_pose()[0][2] > 1.00:
            obj.set_world_pose(
            position=[
                obj.get_world_pose()[0][0],
                obj.get_world_pose()[0][1],
                random.uniform(0.80,0.90)
                
            ],
            orientation = obj.get_world_pose()[1]
        )
        for _ in range(20):
            simulation_context.step(render=True)
    print("Reset the object's Z position!")
            


def find_robot(robot_path):
    """Check if the robot exists in the scene."""
    if is_prim_path_valid(robot_path):
        print(f"Robot found at: {robot_path}")
    else:
        print(f"Robot not found at: {robot_path}")
        exit(1)

def initialize_robot(robot_path:str,initial_joint_positions:np.array,stage,simulation_context):
    """Initialize the robot articulation."""

    robot = Articulation(prim_path=robot_path)
    robot.initialize()

    # launch self_collision
    robot.set_enabled_self_collisions(True)
    # print(f"Self-collision enabled: {robot.get_enabled_self_collisions()}")
    
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    print("Available DOF Names:", robot.dof_names)

    complete_joint_positions = robot.get_joint_positions()
    setting_joint_positions = initial_joint_positions
    complete_joint_positions[:6] = setting_joint_positions
    robot.set_joint_positions(complete_joint_positions)
    for _ in range(5):
        simulation_context.step(render=True)

    from modules.control import set_joint_stiffness_damping
    for dof in robot.dof_names[:7]:
        set_joint_stiffness_damping(stage, find_joint_prim_path_by_name(dof), stiffness=10000.0, damping=1000.0)
    # for dof in robot.dof_names[10:12]: #skip 'left_outer_finger_joint', 'right_outer_finger_joint', 'left_inner_finger_pad_joint', 'right_inner_finger_pad_joint'
    #     set_joint_stiffness_damping(stage, find_joint_prim_path_by_name(dof), stiffness=10000.0, damping=1000.0)
    return robot

def reset_robot_pose(robot,simulation_context):
    complete_joint_positions = robot.get_joint_positions()
    setting_joint_positions = np.array([0, -1.447, 0.749 + random.uniform(-0.087, 0.087), 
                                        -0.873 + random.uniform(-0.087, 0.087), # give the last three joints a random rotation of -5 - 5 degrees
                                        -1.571 + random.uniform(-0.087, 0.087), 
                                        0 + random.uniform(-0.087,0.087)])
    complete_joint_positions[:6] = setting_joint_positions
    robot.set_joint_positions(complete_joint_positions)
    for _ in range(10):
        simulation_context.step(render=True)

def initialize_simulation_context() -> SimulationContext:
    """Initialize and reset the simulation context."""
    simulation_context = SimulationContext()
    while simulation_context.is_simulating():
        time.sleep(0.1)
    simulation_context.initialize_physics()

    simulation_context.reset()
    return simulation_context



def initialize_world(dt: float = 1.0 / 60.0) -> World:
    """Initialize and reset the Isaac Sim world environment."""
    world = World(stage_units_in_meters=1.0)

    world.physics_dt = dt
    
    world.reset()

    return world



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

    return camera


def rgb_and_depth(camera,simulation_context):
    
    while camera.get_depth() is None:
        simulation_context.step(render = True)
    
    depth = camera.get_depth()
    color = camera.get_rgba()[:, :, :3]

    # print(f"RGB Data shape: {color.shape}")
    # print(f"Depth Data shape: {depth.shape}")
    # print("Depth min:", np.min(depth), "max:", np.max(depth))

    # print("camera local pose:",camera.get_local_pose())

    if color is None or depth is None:
        raise RuntimeError("Failed to retrieve RGB or Depth data.")
    data_dict = {
        "rgb": color,
        "depth": depth
    }
    return data_dict

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
    
    # # Save Depth data as normalized grayscale image
    # depth_data = data_dict["depth"]
    # depth_normalized = ((depth_data - np.min(depth_data)) / np.ptp(depth_data) * 255).astype(np.uint8)
    # depth_image = Image.fromarray(depth_normalized)
    # depth_image.save(os.path.join(output_dir, f"{camera_key}_depth_image.png"))
    # print(f"Depth image saved to {os.path.join(output_dir, f'{camera_key}_depth_image.png')}")
    
    # # Save Depth data as NumPy file
    # np.save(os.path.join(output_dir, f"{camera_key}_depth_data.npy"), depth_data)
    # print(f"Depth data saved to {os.path.join(output_dir, f'{camera_key}_depth_data.npy')}")
