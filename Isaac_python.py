"""Launch the simulation_app first."""

from omni.isaac.lab.app import AppLauncher
# launch omniverse app
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""
import sys
import signal
#from pxr import Usd
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.core import World 
from omni.isaac.core.utils.stage import open_stage
from omni.isaac.core.utils.prims import get_prim_at_path
#from omni.isaac.core.articulations import ArticulationController
#from omni.isaac.core.utils.types import ArticulationAction

def handle_signal(signum, frame):
    simulation_context.stop()
    print("Simulation completed.")
    simulation_app.close()
    sys.exit(0)


if __name__ == "__main__":
    
    # Load the saved USD scene
    usd_file_path = "./hello_Isaac.usd"  
    open_stage(usd_path=usd_file_path)

    # Create a World instance
    world = World()
    world.reset()

    # Locate robot in the scene
    robot_path = "/ur10e"  
    robot = get_prim_at_path(robot_path)

    if robot.IsValid():
        print(f"Found robot at: {robot_path}")
    else:
        print(f"Robot not found at: {robot_path}")

    #robot_controller = ArticulationController(robot)
    #current_positions = robot_controller.get_joint_positions()
    #print(current_positions)

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
            
            



