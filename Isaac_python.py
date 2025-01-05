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
import sys
import signal
#from pxr import Usd
from omni.isaac.lab.sim import SimulationContext
# from omni.isaac.urdf import import_urdf
# import_urdf(file_path="./ur10e_with_gripper.urdf", prim_path="/ur10e", enable_collision=True)
# print("aaa")
# exit()
from omni.isaac.core import World  # type: ignore
from omni.isaac.core.utils.stage import open_stage # type: ignore
from omni.isaac.core.utils.prims import get_prim_at_path # type: ignore
from omni.isaac.motion_generation.motion_policy_interface import MotionPolicy # type: ignore
from omni.isaac.motion_generation.articulation_motion_policy import ArticulationMotionPolicy # type: ignore
from omni.isaac.core.articulations import Articulation # type: ignore
from omni.isaac.core.utils.prims import is_prim_path_valid # type: ignore

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

    # Joints States
    # get_joint_info(robot)

    # Get global pose of end
    
    
    # get simulation context
    simulation_context = SimulationContext()
    
    # reset and play simulation
    simulation_context.reset()

    
    # launch simulation
    # try:
    while True:

        # >>>>>>>>>> Add how you control your robot here


        # <<<<<<<<<<

        simulation_context.step()
        signal.signal(signal.SIGINT, handle_signal)
            
          
   


