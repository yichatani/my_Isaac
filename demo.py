"""Launch the simulation application."""
from omni.kit.app import get_app
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

def enable_extensions():
    extension_manager = get_app().get_extension_manager()
    extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)
    extension_manager.set_extension_enabled("omni.physx", True)
enable_extensions()

"""Rest everything follows."""
# Import necessary libraries
import os
import sys
import signal
import numpy as np
from omni.isaac.core import World # type: ignore
from omni.isaac.core.utils.stage import open_stage # type: ignore
from omni.isaac.core.prims import RigidPrim # type: ignore
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore
from omni.isaac.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver # type: ignore
from omni.isaac.core.simulation_context import SimulationContext # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path # type: ignore
from pxr import UsdPhysics, Sdf
#from omni.isaac.core.utils.collision import check_collision # type: ignore


# def enable_collision_for_prim(prim_path):
#     prim = get_prim_at_path(prim_path)
#     if not UsdPhysics.CollisionAPI.Get(prim):
#         UsdPhysics.CollisionAPI.Apply(prim)


# def enable_collision(robot_path):
#     robot_prim = get_prim_at_path(robot_path)
#     for link in robot_prim.GetChildren():
#         collision_api = UsdPhysics.CollisionAPI.Apply(link)
#         if collision_api:
#             print(f"Collision enabled for {link.GetName()}")
#         else:
#             print(f"Failed to enable collision for {link.GetName()}")

# def check_for_collisions(simulation_context):
#     """Custom function to detect collisions in the simulation."""
#     collision_detected = False
#     contacts = simulation_context._physics_context.get_physx_contact_report()
#     if contacts:
#         for contact in contacts:
#             print(f"Collision detected between {contact['prim0']} and {contact['prim1']}")
#             collision_detected = True
#     return collision_detected

# def enable_self_collision(robot_path):
#     """Enable self-collision for all links of a robot."""
#     robot_prim = get_prim_at_path(robot_path)
#     if not robot_prim:
#         print(f"Robot prim not found at path: {robot_path}")
#         return

#     for link in robot_prim.GetChildren():
#         # Check if the link already has PhysicsCollisionAPI
#         if not UsdPhysics.CollisionAPI.Has(link):
#             UsdPhysics.CollisionAPI.Apply(link)
#         # Set self-collision enabled for each link
#         collision_api = UsdPhysics.CollisionAPI(link)
#         if collision_api:
#             collision_api.GetOrCreateAttribute("physics:enableSelfCollision", Sdf.ValueTypeNames.Bool).Set(True)
#             print(f"Enabled self-collision for {link.GetName()}")


def handle_signal(signum, frame):
    """Handle SIGINT for clean exit."""

    print("Simulation interrupted. Exiting...")
    if 'simulation_context' in globals():
        simulation_context.stop()
    if 'simulation_app' in globals():
        simulation_app.close()
    sys.exit(0)

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
    return robot

def initialize_simulation_context():
    """Initialize and reset the simulation context."""
    simulation_context = SimulationContext()
    simulation_context.initialize_physics()
    simulation_context.reset()
    return simulation_context

def get_target_pose():
    """Define the target pose for the end effector."""
    # target_position = np.array([0.864, 0.274, 0.962])  # (x, y, z)
    # target_orientation = np.array([0.707, -0.707, 0, 0])  # Quaternion (qx, qy, qz, qw)
    target_position = np.array([0.5, 0.0, 0.5])  # (x, y, z)
    target_orientation = np.array([0, 0, 0, 1])  # (qx, qy, qz, qw)
    return target_position, target_orientation

def setup_kinematics_solver(robot, yaml_path, urdf_path):
    """Set up the kinematics solver using Lula."""
    kinematics_solver = LulaKinematicsSolver(
        robot_description_path=yaml_path,
        urdf_path=urdf_path
    )
    return kinematics_solver

def interpolate_joint_positions(start_positions, target_positions, steps):
    """
    interpolate
    """
    interpolated_positions = np.linspace(start_positions, target_positions, steps)
    return interpolated_positions

def main():
    # File paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    usd_file_path = os.path.join(ROOT_DIR, "hello_Isaac.usd")
    urdf_path = os.path.join(ROOT_DIR, "ur10e_with_gripper.urdf")
    yaml_path = os.path.join(ROOT_DIR, "ur10e_with_gripper_updated.yaml")
    robot_path = "/ur10e"

    # Open the stage
    open_stage(usd_path=usd_file_path)

    # Enable robot collision
    # enable_collision(robot_path)
    # enable_collision_for_prim()

    # Initialize the world and simulation context
    # world = World()
    simulation_context = initialize_simulation_context()

    # Locate and initialize the robot
    robot = initialize_robot(robot_path)
    complete_joint_positions = np.zeros(12)
    print(f"Joint Positions to Set: {complete_joint_positions}")


    # Set up the kinematics solver
    kinematics_solver = setup_kinematics_solver(robot, yaml_path, urdf_path)

    # Get the end effector frame name
    end_effector_frame_name = "tool0"
    frame_names = kinematics_solver.get_all_frame_names()
    if end_effector_frame_name not in frame_names:
        print(f"Invalid end effector frame name: {end_effector_frame_name}")
        exit(1)

    # Set up the articulation kinematics solver
    articulation_kinematics_solver = ArticulationKinematicsSolver(
        robot_articulation=robot,
        kinematics_solver=kinematics_solver,
        end_effector_frame_name=end_effector_frame_name
    )

    # Main simulation loop
    signal.signal(signal.SIGINT, handle_signal)  # Graceful exit on Ctrl+C
    while True:
        # Define target pose
        target_position, target_orientation = get_target_pose()

        
        print("aaaaaa")
        # Perform inverse kinematics (IK)
        joint_results, success = articulation_kinematics_solver.compute_inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation
        )
        print("bbbbbb")
        


        # Check IK result
        if success:
            joint_positions = joint_results.joint_positions
            print(f"IK Solved! Joint positions: {joint_positions}")
            complete_joint_positions[:len(joint_positions)] = joint_positions


            current_joint_positions = robot.get_joint_positions()
            steps = 100 # decide  how smooth of the path
            trajectory = interpolate_joint_positions(current_joint_positions,complete_joint_positions,steps)

            # print(current_joint_positions)
            # print("aaaaaa")
            # print(complete_joint_positions)
            # print("bbbbbb")
            # print(trajectory)
            #exit()

            # for step, joint_position in enumerate(trajectory):
            #     robot.set_joint_positions(joint_position)
            #     simulation_context.step(render=True)
            #     if check_for_collisions(simulation_context):
            #         print(f"Collision detected at step {step}. Stopping simulation.")
            #         break



            # complete_joint_positions[len(joint_positions):] = gripper_joint_positions
            robot.set_joint_positions(trajectory[1])
        else:
            print("IK Failed! Cannot reach the target pose.")




        # Step simulation
        simulation_context.step(render = True)

if __name__ == "__main__":
    main()
