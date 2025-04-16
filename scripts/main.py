"""Ignore warnings"""
import warnings
warnings.filterwarnings("ignore")

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
import yaml
import signal
import torch
from matplotlib import pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core.utils.stage import open_stage # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.core import World # type: ignore
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from modules.grasp_generator import any_grasp
from modules.control import control_gripper,finger_angle_to_width, control_robot_by_policy
from modules.initial_set import initialize_robot, initialize_simulation_context,initial_camera,reset_robot_pose, rgb_and_depth,reset_obj_pose,reset_obj_z
from modules.record_data import create_episode_file, observing,save_camera_data
from modules.motion_planning import planning_grasp_path
from inference_policy.inference import inference_policy
from Pre_trained_graspnet.inference import pretrained_graspnet

### file_paths
usd_file_path = os.path.join(ROOT_DIR, "../ur10e_grasp_set.usd")
mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
urdf_path = os.path.join(ROOT_DIR, "../urdf/ur10e_gripper.urdf")
yaml_path = kinematics_config_dir + "/universal_robots/ur10e/rmpflow/ur10e_robot_description.yaml"

### prim paths
with open(ROOT_DIR + '/prim_config.yaml', 'r') as file:
    config = yaml.safe_load(file)
robot_path = config['robot_path']
camera_paths = config['camera_paths']
obj_prim_paths = [
    item['path'] for item in config['obj_prim_paths'] if item.get('enabled', False)
]

def handle_signal(signum, frame)-> None:
    """Handle SIGINT for clean exit."""

    print("Simulation interrupted. Exiting...")
    if 'simulation_context' in globals():
        simulation_context.stop() # type: ignore
    if 'simulation_app' in globals():
        simulation_app.close()
    sys.exit(0)

############ main
def main(is_policy=False, self_trained_model=None) -> None:
    """Main function to run the simulation.
    Args:
        is_policy (bool): Flag to indicate if using policy for control.
        self_trained_model (bool): Flag to indicate if using a self-trained model.
    """
    # Open the stage
    stage = open_stage(usd_path=usd_file_path)
    # Initialize the world and simulation context
    simulation_context = initialize_simulation_context()
    
    # Initial robot
    robot = initialize_robot(robot_path)
    for _ in range(1):
        simulation_context.step(render=True)
    # Initial ArticulationKinematicsSolver
    LulaKSolver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    # print("KSolver get_all_frame_names:",LulaKSolver.get_all_frame_names())
    AKSolver = ArticulationKinematicsSolver(robot,LulaKSolver,"tool0")

    global camera_paths
    sensor = initial_camera(camera_paths["sensor"],60,(1920,1080))
    record_camera_dict = {
        "in_hand": initial_camera(camera_paths["in_hand"],60,(448,448)),
        "up": initial_camera(camera_paths["up"],60,(448,448)),
        "front": initial_camera(camera_paths["front"],60,(448,448))
    }

    # Main simulation loop #
    signal.signal(signal.SIGINT, handle_signal)  # Graceful exit on Ctrl+C
    episode_count = 0
    # while True:
    for _ in range(300):
                    
        if is_policy:
            reset_obj_pose(obj_prim_paths,simulation_context)
            reset_robot_pose(robot,simulation_context)
            data_sample = None
            for _ in range(100):
                if _ == 0:
                    data_sample = observing(robot,record_camera_dict,simulation_context,data_sample)
                else:
                    actions = inference_policy(data_sample,obs_steps=3,action_steps=6)
                    data_sample = control_robot_by_policy(robot,record_camera_dict,actions,simulation_context,data_sample)

        else:
            reset_obj_pose(obj_prim_paths,simulation_context)
            for _ in range(10):
                reset_robot_pose(robot,simulation_context)
                data_dict = rgb_and_depth(sensor,simulation_context)
                # any_data_dict = any_grasp(data_dict)
                if self_trained_model is not None:
                    assert self_trained_model=="1billion.tar" or "mega.tar", "self_trained_model invalid"
                    print(f"<<<Using self-trained model: {self_trained_model}>>>")
                    any_data_dict = pretrained_graspnet(data_dict, chosen_model=self_trained_model)
                else:
                    print(f"<<<Using AnyGrasp>>>")
                    any_data_dict = any_grasp(data_dict)
                if any_data_dict is False:
                    break
                episode_path = create_episode_file(record_camera_dict,height=448,width=448)
                planning_grasp_path(robot,record_camera_dict, any_data_dict,AKSolver,simulation_context,episode_path)
                reset_obj_z(obj_prim_paths,simulation_context)
                if episode_count % 10 == 0:
                    torch.cuda.empty_cache()
                
                print(f"Completed {episode_count} episodes.")
                episode_count += 1
                # Clean
                torch.cuda.empty_cache()  # clean GPU

if __name__ == "__main__":
    
    main(is_policy = True)
    # main(is_policy = False, self_trained_model="1billion.tar")
    

    
