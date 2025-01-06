"""Launch the simulation_app first."""

from omni.isaac.lab.app import AppLauncher
from omni.kit.app import get_app
# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app
# activate extension
extension_manager = get_app().get_extension_manager()
extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)

"""Rest everything follows."""
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
import signal
import numpy as np
#from pxr import Usd
from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.urdf import import_urdf
# import_urdf(file_path="./ur10e_with_gripper.urdf", prim_path="/ur10e", enable_collision=True)
# print("aaa")
# exit()
from omni.isaac.core import World  # type: ignore
from omni.isaac.core.prims import RigidPrim # type: ignore
from omni.isaac.core.utils.stage import open_stage # type: ignore
#from omni.isaac.core.utils.prims import get_prim_at_path # type: ignore
from omni.isaac.motion_generation.motion_policy_interface import MotionPolicy # type: ignore
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy # type: ignore
#from omni.isaac.core.controllers import ArticulationController # type: ignore
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver # type: ignore

def Breakpoints():
    print("everything ok!")
    exit()

def find_robot(robot_path):
    if is_prim_path_valid(robot_path):
        print(f"Found robot at: {robot_path}")
    else:
        print(f"Robot not found at: {robot_path}")

def handle_signal(signum, frame):
    simulation_context.stop()
    print("Simulation completed.")
    simulation_app.close()
    sys.exit(0)

def get_joint_info(robot):
    joint_states = robot.get_joints_state()
    joint_positions = joint_states.positions
    # Joint positions
    # joint_positions = robot.get_joint_positions()
    # print("Joints Positions:", joint_positions)
    joint_velocities = joint_states.velocities
    joint_efforts = joint_states.efforts
    joint_names = robot.dof_names
    # Joints Info
    for name, position, velocity, effort in zip(joint_names, joint_positions, joint_velocities, joint_efforts):
        print(f"Joint Name: {name}, Position: {position}, Velocity: {velocity}, Effort: {effort}")


if __name__ == "__main__":
    
    # Load the saved USD scene
    usd_file_path = "./hello_Isaac.usd"  
    open_stage(usd_path=usd_file_path)

    # Create a World instance
    world = World()
    world.reset()

    # Locate robot in the scene
    robot_path = "/ur10e"
    find_robot(robot_path)
    robot = Articulation(prim_path=robot_path)
    # Must Initialize!
    robot.initialize()
    
    # Create MotionPolicy and ArticulationMotionPolicy
    mp = MotionPolicy()
    amp = ArticulationMotionPolicy(robot,mp)

    # LulaKinematicSolver
    kinematics_solver = LulaKinematicsSolver(
        robot_description_path=os.path.join(ROOT_DIR, "ur10e_with_gripper_updated.yaml"),
        urdf_path=os.path.join(ROOT_DIR, "ur10e_with_gripper.urdf")
    )

    # Get global pose of end
    robotiqpad_R_path = "/ur10e/right_inner_finger_pad"
    robotiqpad_L_path = "/ur10e/left_inner_finger_pad"
    
    robotiqpad_R = RigidPrim(prim_path = robotiqpad_R_path)
    robotiqpad_L = RigidPrim(prim_path = robotiqpad_L_path)
    robotiqpad_R_world = robotiqpad_R.get_world_pose()
    robotiqpad_L_world = robotiqpad_L.get_world_pose()

    # print("right:",robotiqpad_R_world)
    # print("left:",robotiqpad_L_world)

    end_pose_translation = (robotiqpad_R_world[0]+ robotiqpad_L_world[0])/2
    end_pose_rotation = robotiqpad_R_world[1]
    # print("end_pose_translation:",end_pose_translation)
    # print("end_pose_rotation:",end_pose_rotation)

    ######Get data from Anygrasp to know the end
    grasp_width = None
    grasp_translation_camera = None
    grasp_rotation_camera = None  #should be a matrix



    # set pose
    target_position = np.array([0.5, 0.0, 0.5])  # (x, y, z)
    target_orientation = np.array([0, 0, 0, 1])  # (qx, qy, qz, qw)
    # IK
    end_effector_frame_name = "tool0"  # end_Frame
    frame_names = kinematics_solver.get_all_frame_names()
    print("Available frame names:", frame_names)
    assert end_effector_frame_name in frame_names, "End effector frame name is invalid!"


    articulation_kinematics_solver = ArticulationKinematicsSolver(robot_articulation=robot, 
                                                    kinematics_solver=kinematics_solver, 
                                                    end_effector_frame_name=end_effector_frame_name)
    # caculate joints
    joint_positions, success = articulation_kinematics_solver.compute_inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation
    )


    # get simulation context
    simulation_context = SimulationContext()
    
    # Initialize physics
    simulation_context.initialize_physics()  

    # reset and play simulation
    simulation_context.reset()

    # launch simulation
    # try:
    while True:

        # >>>>>>>>>> Add how you control your robot here
        if success:
            print("Find successfully:", joint_positions)
            # apply
            robot.set_joint_positions(joint_positions)
        else:
            print("Can't reach, please check again!!")
        # <<<<<<<<<<

        simulation_context.step()
        signal.signal(signal.SIGINT, handle_signal)
            







# """Launch the simulation_app first."""
# from omni.isaac.lab.app import AppLauncher
# from omni.kit.app import get_app
# # launch omniverse app
# app_launcher = AppLauncher(headless=False)
# simulation_app = app_launcher.app
# # activate extension
# extension_manager = get_app().get_extension_manager()
# extension_manager.set_extension_enabled("omni.isaac.motion_generation", True)

# """Rest everything follows."""
# import sys
# import signal
# import numpy as np
# from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.core import World  # type: ignore
# from omni.isaac.core.prims import RigidPrim  # type: ignore
# from omni.isaac.core.utils.stage import open_stage  # type: ignore
# from omni.isaac.core.utils.prims import get_prim_at_path  # type: ignore
# from omni.isaac.motion_generation.motion_policy_interface import MotionPolicy  # type: ignore
# from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy  # type: ignore
# from omni.isaac.core.controllers import ArticulationController  # type: ignore
# #from omni.isaac.core.controllers import BaseController
# from omni.isaac.core.articulations import Articulation  # type: ignore
# from omni.isaac.core.utils.prims import is_prim_path_valid  # type: ignore

# help(ArticulationMotionPolicy)

# exit()

# def Breakpoints():
#     print("everything ok!")
#     exit()

# def find_robot(robot_path):
#     if is_prim_path_valid(robot_path):
#         print(f"Found robot at: {robot_path}")
#     else:
#         print(f"Robot not found at: {robot_path}")

# def handle_signal(signum, frame):
#     simulation_context.stop()
#     print("Simulation completed.")
#     simulation_app.close()
#     sys.exit(0)

# def get_joint_info(robot):
#     joint_states = robot.get_joints_state()
#     joint_positions = joint_states.positions
#     joint_velocities = joint_states.velocities
#     joint_efforts = joint_states.efforts
#     joint_names = robot.dof_names
#     for name, position, velocity, effort in zip(joint_names, joint_positions, joint_velocities, joint_efforts):
#         print(f"Joint Name: {name}, Position: {position}, Velocity: {velocity}, Effort: {effort}")

# if __name__ == "__main__":
    
#     # Load the saved USD scene
#     usd_file_path = "./hello_Isaac.usd"  
#     open_stage(usd_path=usd_file_path)

#     # Create a World instance
#     world = World()
#     world.reset()

#     # Locate robot in the scene
#     robot_path = "/ur10e"
#     find_robot(robot_path)
#     robot = Articulation(prim_path=robot_path)

#     # Must Initialize!
#     robot.initialize()
    
#     # Create MotionPolicy and ArticulationMotionPolicy
#     mp = MotionPolicy()
#     amp = ArticulationMotionPolicy(robot, mp)

#     # Joints States
#     # get_joint_info(robot)

#     # Get global pose of end
#     robotiqpad_R_path = "/ur10e/right_inner_finger_pad"
#     robotiqpad_L_path = "/ur10e/left_inner_finger_pad"
    
#     robotiqpad_R = RigidPrim(prim_path=robotiqpad_R_path)
#     robotiqpad_L = RigidPrim(prim_path=robotiqpad_L_path)
#     robotiqpad_R_world = robotiqpad_R.get_world_pose()
#     robotiqpad_L_world = robotiqpad_L.get_world_pose()

#     print("right:", robotiqpad_R_world)
#     print("left:", robotiqpad_L_world)

#     end_pose_translation = (robotiqpad_R_world[0] + robotiqpad_L_world[0]) / 2
#     end_pose_rotation = robotiqpad_R_world[1]
#     print("end_pose_translation:", end_pose_translation)
#     print("end_pose_rotation:", end_pose_rotation)

#     ######Get data from Anygrasp to know the end
#     grasp_width = None
#     grasp_translation_camera = None
#     grasp_rotation_camera = None  # should be a matrix

#     # get simulation context
#     simulation_context = SimulationContext()
    
#     # reset and play simulation
#     simulation_context.reset()

#     # Create a controller for the robot
#     controller = ArticulationController()
#     controller.initialize(robot)

#     signal.signal(signal.SIGINT, handle_signal)

#     # launch simulation
#     while True:
#         # >>>>>>>>>> Add how you control your robot here
#         ######Define target pose for the end effector
#         target_position = np.array([0.5, 0.0, 0.5])  # Target position
#         target_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Target orientation (unit quaternion)
        
#         joint_positions = amp.compute_joint_positions(target_position, target_orientation)
        
#         # Set the joint pose for the controller
#         # robot.set_joint_positions(joint_positions)
#         #controller.set_target_pose(robot, target_position, target_orientation)
#         controller.set_joint_positions(robot, joint_positions)
        
#         # <<<<<<<<<<

#         simulation_context.step()
        
   


