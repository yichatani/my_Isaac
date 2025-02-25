"""Launch the simulation application."""
from omni.kit.app import get_app # type: ignore
from omni.isaac.kit import SimulationApp # type: ignore
simulation_app = SimulationApp({"headless": False})

def enable_extensions():
    extension_manager = get_app().get_extension_manager()
    extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)
    extension_manager.set_extension_enabled("omni.physx", True)
    extension_manager.set_extension_enabled("omni.isaac.dynamic_control", True)
enable_extensions()

"""Rest everything follows."""
# Import necessary libraries
import os
import sys
import signal
import numpy as np
import torch
import time
import threading
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core.utils.stage import open_stage # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from modules.grasp_generator import any_grasp
from modules.control import control_gripper
from modules.initial_set import initialize_robot, initialize_simulation_context,initial_camera,find_robot, rgb_and_depth
from modules.record_data import recording
from modules.motion_planning import planning_grasp_path
# import omni
# omni.timeline.get_timeline_interface().play()
# print("Every thing ok!")
# exit()

### Paths
usd_file_path = os.path.join(ROOT_DIR, "../ur10e_grasp_set.usd")
mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
# print("kinematics_config_dir:",kinematics_config_dir)
# urdf_path = kinematics_config_dir + "/universal_robots/ur10e/ur10e.urdf"
urdf_path = os.path.join(ROOT_DIR, "../urdf/ur10e_gripper.urdf")
yaml_path = kinematics_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_robot_description.yaml"
# yaml_path = os.path.join(ROOT_DIR, "../ur10e_description.yaml")
# rrt_config_path = os.path.join(ROOT_DIR, "../controller/rrt_config.yaml")

### Prim path
robot_path = "/ur10e"
camera_path = "/ur10e/tool0/Camera"
# tool0_path = "/ur10e/tool0"
# baselink_path = "/ur10e/base_link"
###

camera_paths = {
    "sensor": "/ur10e/tool0/Camera",
    "in_hand": "/ur10e/tool0/in_hand",
    "up": "/World/up",
    "front": "/World/front"
}

recording_event = threading.Event()
stop_event = threading.Event()

def handle_signal(signum, frame):
    """Handle SIGINT for clean exit."""

    print("Simulation interrupted. Exiting...")
    if 'simulation_context' in globals():
        simulation_context.stop() # type: ignore
    if 'simulation_app' in globals():
        simulation_app.close()
    sys.exit(0)

############ main
def main():
    
    ##############
    # Initialize #
    ##############
    ############ Open the stage
    open_stage(usd_path=usd_file_path)
    # Initialize the world and simulation context
    simulation_context = initialize_simulation_context()

    ############ Initial robot
    robot = initialize_robot(robot_path)
    complete_joint_positions = robot.get_joint_positions()
    setting_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    putting_joint_positions = np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0])
    complete_joint_positions[:6] = setting_joint_positions
    robot.set_joint_positions(complete_joint_positions)
    find_robot(robot_path)
    for _ in range(50):
        simulation_context.step(render=True)
        # recording_event.set()
        # if not recording_event.is_set():
        #     recording_event.set()
    
    ############ Initial ArticulationKinematicsSolver
    LulaKSolver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    # print("KSolver get_all_frame_names:",LulaKSolver.get_all_frame_names())
    AKSolver = ArticulationKinematicsSolver(robot,LulaKSolver,"tool0")

    ########################
    # Main simulation loop #
    ########################
    signal.signal(signal.SIGINT, handle_signal)  # Graceful exit on Ctrl+C

    while True:
        global camera_paths
        sensor = initial_camera(camera_paths["sensor"],60,(1920,1080))
        in_hand_cam = initial_camera(camera_paths["in_hand"],60,(640,480))
        up_cam = initial_camera(camera_paths["up"],60,(640,480))
        front_cam = initial_camera(camera_paths["front"],60,(640,480))

        record_camera_dict = {
            "in_hand": in_hand_cam,
            "up": up_cam,
            "front": front_cam
        }

        record_thread = threading.Thread(target=recording, args=(robot, record_camera_dict, simulation_context, recording_event, stop_event,))
        record_thread.start()

        # get rgb and depth data for processing
        data_dict = rgb_and_depth(sensor,simulation_context)
        
        # save_camera_data(data_dict)
        any_data_dict = any_grasp(data_dict)
        complete_joint_positions = control_gripper(robot, 0.14,any_data_dict["width"],complete_joint_positions,simulation_context,recording_event)
        
        planning_grasp_path(robot,any_data_dict,AKSolver,simulation_context,recording_event)
        
        stop_event.set()
        record_thread.join()
        print("Main thread: Recording thread stopped.")

        # Clean
        torch.cuda.empty_cache()  # clean GPU

    

if __name__ == "__main__":
    
    main()

    
