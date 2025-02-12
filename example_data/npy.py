import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 .npy 文件
depth_data = np.load(os.path.join(ROOT_DIR, "depth_1.npy"))

def use_plt(depth_data):
    # 显示深度图
    plt.imshow(depth_data, cmap='viridis')  # 可以选择其他色图，如 'plasma', 'gray'
    plt.colorbar(label="Depth")
    plt.title("Depth Map")
    plt.show()


def use_cv2(depth_data):
    # 将深度值归一化为 0-255（方便可视化）
    depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # 显示深度图
    cv2.imshow("Depth Map", depth_normalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def vis_3d(depth_data):
    # 创建网格坐标
    x = np.arange(depth_data.shape[1])
    y = np.arange(depth_data.shape[0])
    x, y = np.meshgrid(x, y)

    # 3D 可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, depth_data, cmap='viridis')
    plt.title("3D Depth Visualization")
    plt.show()

if __name__ == "__main__":
    #use_cv2(depth_data)
    use_plt(depth_data)
    #vis_3d(depth_data)
