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
from matplotlib import pyplot as plt
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core.utils.stage import open_stage # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.core import World # type: ignore
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from modules.grasp_generator import any_grasp
from modules.control import control_gripper,finger_angle_to_width, start_force_control_gripper, stop_force_control_gripper
from modules.initial_set import initialize_robot, initialize_simulation_context,initial_camera,reset_robot_pose, rgb_and_depth,reset_obj_position
from modules.record_data import create_episode_file
from modules.motion_planning import planning_grasp_path

### Paths
usd_file_path = os.path.join(ROOT_DIR, "../ur10e_grasp_set.usd")
mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
urdf_path = os.path.join(ROOT_DIR, "../urdf/ur10e_gripper.urdf")
yaml_path = kinematics_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_robot_description.yaml"

### Prim path
robot_path = "/ur10e"
camera_paths = {
    "sensor": "/ur10e/tool0/Camera",
    "in_hand": "/ur10e/tool0/in_hand",
    "up": "/World/up",
    "front": "/World/front"
}
# test
obj_prim_path = [
    "/rubiks_cube",
    # "/Lemon_01",
    # "/_02_master_chef_can",
    "/_09_gelatin_box",
    "/Android_Figure_Panda",
    "/nvidia_cube",
    #"/SM_Mug_A2",
    "/_05_tomato_soup_can",
    "/_11_banana",
    "/Office_Depot_Canon",
    #"/_10_potted_meat_can",
    "/_08_pudding_box"
]

# recording_event = threading.Event()
# stop_event = threading.Event()

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
    
    # Initialize #
    # Open the stage
    stage = open_stage(usd_path=usd_file_path)
    # Initialize the world and simulation context
    simulation_context = initialize_simulation_context()
    # world = World()
    # get_all_prim_paths(stage)

    # print(dir(simulation_context))

    # exit()
    # world = World()
    # reset_obj_position(obj_prim_path,world)


    # Initial robot
    robot = initialize_robot(robot_path)
    # reset_robot_pose(robot)
    # robot_go_home(robot)
    for _ in range(1):
        simulation_context.step(render=True)
    # find_robot(robot_path)
    
    # Initial ArticulationKinematicsSolver
    LulaKSolver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    # print("KSolver get_all_frame_names:",LulaKSolver.get_all_frame_names())
    AKSolver = ArticulationKinematicsSolver(robot,LulaKSolver,"tool0")

    global camera_paths
    sensor = initial_camera(camera_paths["sensor"],60,(1920,1080))
    in_hand_cam = initial_camera(camera_paths["in_hand"],60,(448,448))
    up_cam = initial_camera(camera_paths["up"],60,(448,448))
    front_cam = initial_camera(camera_paths["front"],60,(448,448))
    record_camera_dict = {
        "in_hand": in_hand_cam,
        "up": up_cam,
        "front": front_cam
    }

    # Main simulation loop #
    signal.signal(signal.SIGINT, handle_signal)  # Graceful exit on Ctrl+C
    episode_count = 0
    while True:
        
        reset_obj_position(obj_prim_path,simulation_context)

        for _ in range(10):
            # stop_event.clear()
            # record_thread = threading.Thread(target=recording, args=(robot, record_camera_dict, simulation_context, recording_event, stop_event,))
            # record_thread.start()
            reset_robot_pose(robot,simulation_context)
            episode_path = create_episode_file(record_camera_dict,height=448,width=448)

            data_dict = rgb_and_depth(sensor,simulation_context)

            # save_camera_data(data_dict)
            any_data_dict = any_grasp(data_dict)
            if any_data_dict is False:
                # reset_obj_position(obj_prim_path)
                # for _ in range(50):
                #     simulation_context.step(render=True)
                break
            complete_joint_positions = robot.get_joint_positions()
            
            # start_force_control_gripper(robot)
            # check_data = np.array([])
            # for _ in range(100):
            #     # print(robot.get_joint_positions())
            #     check_data = np.append(check_data, robot.get_joint_positions()[6])
            #     simulation_context.step(render=True)
            # print(check_data)
            # x_values = np.arange(len(check_data))
            # y_values = check_data
            # plt.plot(x_values,y_values)
            # stop_force_control_gripper(robot)
            # exit()

            complete_joint_positions = control_gripper(robot, record_camera_dict, finger_angle_to_width(complete_joint_positions[6]),any_data_dict["width"],
                                                    complete_joint_positions,simulation_context,episode_path,is_record=True)

            # complete_joint_positions = control_gripper(robot, record_camera_dict, finger_angle_to_width(complete_joint_positions[6]),finger_angle_to_width(0.7),
            #                                         complete_joint_positions,simulation_context,episode_path,is_record=False)
            
            # exit()
            
            
            planning_grasp_path(robot,record_camera_dict, any_data_dict,AKSolver,simulation_context,episode_path)
            # robot_go_home(robot)

            if episode_count % 10 == 0:
                torch.cuda.empty_cache()
            
            print(f"Completed {episode_count} episodes.")

            episode_count += 1
            # stop_event.set()
            # record_thread.join()
            # print("Recording thread stopped.")

            # # Clean
            # torch.cuda.empty_cache()  # clean GPU

    

if __name__ == "__main__":
    
    main()

    
