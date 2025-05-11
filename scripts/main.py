"""Ignore warnings"""
import warnings
warnings.filterwarnings("ignore")

"""Launch the simulation application."""
from omni.kit.app import get_app # type: ignore
from omni.isaac.kit import SimulationApp # type: ignore
simulation_app = SimulationApp({
    "headless": False,                          # If need GUI
    "hide_ui": True,                            
    "active_gpu": 0,                            # Set GPU
    "physics_gpu": 0,
    "multi_gpu": False,                         
    "max_gpu_count": None,
    "sync_loads": True,                        
    "width": 1280,                             
    "height": 720,                             
    "window_width": 1440,
    "window_height": 900,
    "display_options": 3094,                    
    "subdiv_refinement_level": 0,               
    "renderer": "RayTracedLighting",            
    "anti_aliasing": 3,                         
    "samples_per_pixel_per_frame": 64,          
    "denoiser": True,                           
    "max_bounces": 4,                           
    "max_specular_transmission_bounces": 6,
    "max_volume_bounces": 4,
    "open_usd": None,
    "livesync_usd": None,
    "fast_shutdown": True,
    "profiler_backend": [],
    })


"""Rest everything follows."""
# Import necessary libraries
import os
import sys
import yaml
import signal
import torch
import time
import numpy as np
from matplotlib import pyplot as plt
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from omni.isaac.core.utils.stage import open_stage, get_current_stage # type: ignore
from omni.isaac.core.utils.extensions import get_extension_path_from_name # type: ignore
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats, euler_angles_to_quats,quats_to_euler_angles # type: ignore
from omni.isaac.core.utils.types import ArticulationAction # type: ignore
from omni.isaac.core.articulations import ArticulationView # type: ignore
from omni.isaac.core import World # type: ignore
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from modules.grasp_generator import any_grasp
from modules.control import control_robot_by_policy
from modules.initial_set import initialize_robot, initialize_simulation_context,initial_camera,reset_robot_pose, \
    rgb_and_depth,reset_obj_pose,check_obj_pose_err,initialize_world
from modules.record_data import observing
from modules.motion_planning import planning_grasp_path
from inference_policy.inference import inference_policy


### file_paths
usd_file_path = os.path.join(ROOT_DIR, "../ur10e_grasp.usd")
mg_extension_path = get_extension_path_from_name("omni.isaac.motion_generation")
kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
# print("kinematics_config_dir:", kinematics_config_dir)
urdf_path = kinematics_config_dir + "/universal_robots/ur10e/ur10e.urdf"
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
    print("Opening stage...")
    open_stage(usd_path=usd_file_path)
    stage = get_current_stage()
    print("Stage opened.")
    # simulation_context = initialize_simulation_context()
    simulation_context = initialize_world(dt=1.0/60.0)
    
    # Initial robot
    initial_joint_positions = np.array([0, -1.447, 0.749, -0.873, -1.571, 0])
    ending_joint_positions = np.array([-0.85, -1.147, 0.549, -0.873, -1.571, 0])
    robot = initialize_robot(robot_path,initial_joint_positions,stage,simulation_context)
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
    for _ in range(400):
                    
        if is_policy:
            reset_obj_pose(obj_prim_paths,simulation_context)
            reset_robot_pose(robot,simulation_context)
            data_sample = None
            data_sample = observing(robot,record_camera_dict,simulation_context,data_sample,obs_steps=4)
            for _ in range(11):
                actions = inference_policy(data_sample,obs_steps=4,action_steps=3)
                joint_actions = []
                for action in actions:
                    T_quat = euler_angles_to_quats(action[3:6])
                    T_joint_state, succ = AKSolver.compute_inverse_kinematics(action[:3], T_quat)
                    if not succ:
                        print("IK failed, skipping")
                        continue
                    joint_action = T_joint_state.joint_positions
                    joint_action = np.concatenate((joint_action, action[6:7]), axis=0)
                    joint_actions.append(joint_action)
                joint_actions = np.stack(joint_actions, axis=0)
                if len(joint_actions) == 0:
                    print("All IKs failed, skipping this step.")
                    break
                assert joint_actions.shape[0] == actions.shape[0], "Mismatch in action step count!"
                data_sample = control_robot_by_policy(robot,record_camera_dict,joint_actions,simulation_context,data_sample,obs_steps=4)

        else:
            reset_obj_pose(obj_prim_paths,simulation_context)
            for _ in range(10):
                if check_obj_pose_err(obj_prim_paths):
                    break
                reset_robot_pose(robot,simulation_context)
                data_dict = rgb_and_depth(sensor,simulation_context)
                if self_trained_model is not None:
                    assert self_trained_model in ["1billion.tar", "mega.tar"], "self_trained_model invalid"
                    print(f"<<<Using self-trained model: {self_trained_model}>>>")
                    from Pre_trained_graspnet.inference import pretrained_graspnet
                    any_data_dict = pretrained_graspnet(data_dict, chosen_model=self_trained_model)
                else:
                    print(f"<<<Using AnyGrasp>>>")
                    any_data_dict = any_grasp(data_dict)
                if not any_data_dict: 
                    break
                plan_succ = planning_grasp_path(robot,record_camera_dict, any_data_dict,AKSolver,simulation_context,
                                    initial_joint_positions,ending_joint_positions)
                if not plan_succ:
                    print("Planning failed, skipping")
                    continue
                if episode_count % 10 == 0:
                    torch.cuda.empty_cache()
                print(f"Completed {episode_count} episodes.")
                episode_count += 1
                # Clean
                torch.cuda.empty_cache()  # clean GPU

if __name__ == "__main__":
    
    main(is_policy = False)
    

    
