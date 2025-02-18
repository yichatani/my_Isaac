import os
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore
from omni.isaac.core.simulation_context import SimulationContext # type: ignore
import omni.replicator.core as rep # type: ignore
from omni.isaac.core.utils.stage import get_current_stage # type: ignore
from pxr import UsdPhysics # type: ignore
from omni.isaac.sensor import Camera # type: ignore

def find_robot(robot_path):
    """Check if the robot exists in the scene."""
    if is_prim_path_valid(robot_path):
        print(f"Robot found at: {robot_path}")
    else:
        print(f"Robot not found at: {robot_path}")
        exit(1)

def initialize_robot(robot_path):
    """Initialize the robot articulation."""

    robot = Articulation(prim_path=robot_path)
    robot.initialize()

    # launch self_collision
    robot.set_enabled_self_collisions(True)
    print(f"Self-collision enabled: {robot.get_enabled_self_collisions()}")
    
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(64)
    print("Available DOF Names:", robot.dof_names)

    print(f"Initial joint positions: {robot.get_joint_positions()}")

    return robot

def initialize_simulation_context():
    """Initialize and reset the simulation context."""
    simulation_context = SimulationContext()
    while simulation_context.is_simulating():
        time.sleep(0.1)
    simulation_context.initialize_physics()

    simulation_context.reset()
    return simulation_context



def initial_camera(camera_path):
    """Initialize Camera"""
    camera = Camera(
        prim_path=camera_path,
        frequency=60,
        resolution=(1920, 1080),
    )

    camera.initialize()
    camera.add_motion_vectors_to_frame()
    camera.add_distance_to_image_plane_to_frame()
    camera = set_camera_parameters(camera)


def rgb_and_depth(camera_path,simulation_context):
    
    print("FFFFF")
    
    camera = Camera(prim_path=camera_path)

    print("DDDDD")

    while camera.get_depth() is None:
        simulation_context.step(render = True)

    print("EEEEE")
    
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

    camera.set_clipping_range(0.01, 1.0e7)

    return camera

    
def save_camera_data(data_dict, output_dir="./output_data"):
    """
    Save RGB and Depth data to files.
    
    Args:
        data_dict (dict): Dictionary with "rgb" and "depth" data.
        output_dir (str): Directory to save the files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save RGB image
    rgb_image = Image.fromarray(data_dict["rgb"])
    rgb_image.save(os.path.join(output_dir, "rgb_image.png"))
    print(f"RGB image saved to {os.path.join(output_dir, 'rgb_image.png')}")
    
    # Save Depth data as normalized grayscale image
    depth_data = data_dict["depth"]
    depth_normalized = ((depth_data - np.min(depth_data)) / np.ptp(depth_data) * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_normalized)
    depth_image.save(os.path.join(output_dir, "depth_image.png"))
    print(f"Depth image saved to {os.path.join(output_dir, 'depth_image.png')}")
    
    # Save Depth data as NumPy file
    np.save(os.path.join(output_dir, "depth_data.npy"), depth_data)
    print(f"Depth data saved to {os.path.join(output_dir, 'depth_data.npy')}")


"""

    Deprecated below.

"""
def initial_camera__(camera_path):
    """Deprecated"""
    rep.new_layer()
    camera = rep.get.prim_at_path(camera_path)
    if not camera:
        raise RuntimeError(f"Camera not found at path: {camera_path}")
    print(f"Using camera at path: {camera_path}")
    # Turn off camera's physics
    camera_prim = get_current_stage().GetPrimAtPath(camera_path)
    physics_api = UsdPhysics.RigidBodyAPI.Apply(camera_prim)
    physics_api.GetRigidBodyEnabledAttr().Set(False)

    render_product = rep.create.render_product(camera_path, resolution=(1920, 1080))
    # Render_product = rep.create.render_product(camera_path)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_annotator = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_annotator.attach([render_product])
    depth_annotator.attach([render_product])

    return rgb_annotator,depth_annotator

def set_camera_parameters__(camera):
    # OpenCV camera matrix and width and height of the camera sensor, from the calibration file
    width, height = 1920, 1080
    # camera_matrix = [[1662.77, 0.0, 970.94], [0.0, 1281.77, 600.37], [0.0, 0.0, 1.0]]
    camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]

    # Pixel size in microns, aperture and focus distance from the camera sensor specification
    # Note: to disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
    pixel_size = 3 * 1e-3   # in mm, 3 microns is a common pixel size for high resolution cameras
    f_stop = 1.8            # f-number, the ratio of the lens focal length to the diameter of the entrance pupil
    focus_distance = 0.6    # in meters, the distance from the camera to the object plane

    # Calculate the focal length and aperture size from the camera matrix
    ((fx,_,cx),(_,fy,cy),(_,_,_)) = camera_matrix
    horizontal_aperture =  pixel_size * width                   # The aperture size in mm
    vertical_aperture =  pixel_size * height
    focal_length_x  = fx * pixel_size
    focal_length_y  = fy * pixel_size
    focal_length = (focal_length_x + focal_length_y) / 2         # The focal length in mm

    # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
    camera.set_focal_length(focal_length / 10.0)                # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_focus_distance(focus_distance)                   # The focus distance in meters
    camera.set_lens_aperture(f_stop * 100.0)                    # Convert the f-stop to Isaac Sim units
    camera.set_horizontal_aperture(horizontal_aperture / 10.0)  # Convert from mm to cm (or 1/10th of a world unit)
    camera.set_vertical_aperture(vertical_aperture / 10.0)

    camera.set_clipping_range(0.05, 1.0e5)

    return camera
