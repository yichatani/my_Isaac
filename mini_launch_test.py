# """Launch the simulation_app first."""

# from omni.isaac.lab.app import AppLauncher
# from omni.kit.app import get_app
# # launch omniverse app
# app_launcher = AppLauncher(headless=False)
# simulation_app = app_launcher.app
# # activate extension
# extension_manager = get_app().get_extension_manager()
# extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)



# 初始化Isaac Sim环境
from omni.isaac.kit import SimulationApp
from omni.kit.app import get_app
simulation_app = SimulationApp({"headless": False})  # 如果需要GUI可设置为 False
# activate extension
extension_manager = get_app().get_extension_manager()
extension_manager.set_extension_enabled("omni.isaac.motion_generation",True)


import numpy as np
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationKinematicsSolver


# 加载环境和机械臂
from omni.isaac.core.utils.stage import open_stage

# 加载USD场景文件
open_stage("./hello_Isaac.usd")  # 替换为您的USD路径

# 获取机械臂
from omni.isaac.core import World
world = World(stage_units_in_meters=1.0)
robot_prim_path = "/ur10e"  # 替换为USD文件中机械臂的路径
robot = Articulation(prim_path=robot_prim_path)
robot.initialize()
# world.add_articulation(robot)

# 启动模拟
world.reset()
simulation_app.update()

# 定义目标末端位姿
target_position = np.array([0.5, 0.0, 0.5])  # 替换为目标末端位置 (x, y, z)
target_orientation = np.array([0, 0, 0, 1])  # 替换为目标末端四元数方向 (qx, qy, qz, qw)

# 初始化逆运动学求解器
end_effector_frame_name = "tool0"  # 替换为末端执行器的Frame名称
kinematics_solver = ArticulationKinematicsSolver(robot_articulation=robot, 
                                                 kinematics_solver=None, 
                                                 end_effector_frame_name=end_effector_frame_name)

# 计算关节位置
joint_positions, success = kinematics_solver.compute_inverse_kinematics(
    target_position=target_position,
    target_orientation=target_orientation
)

if success:
    print("计算成功，关节位置：", joint_positions)
    
    # 应用关节位置到机械臂
    robot.set_joint_positions(joint_positions)
    
    # 更新模拟环境
    for _ in range(100):  # 运行100步模拟以观察机械臂运动
        world.step(render=True)
else:
    print("无法计算目标关节位置，请检查目标末端位姿是否可达。")

# 停止模拟
simulation_app.close()
