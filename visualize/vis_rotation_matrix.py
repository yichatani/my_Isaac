import numpy as np
import open3d as o3d

# angle = np.radians(45)  #  Z 45
# rotation_matrix = np.array([
#     [np.cos(angle), -np.sin(angle), 0],
#     [np.sin(angle), np.cos(angle),  0],
#     [0,             0,              1]
# ])

rotation_matrix_1 = np.array([
    [ 4.85149095e-07, -9.99999979e-01, 2.02789612e-04],
    [-9.99999979e-01, -5.26375266e-07 ,-2.03295272e-04],
    [ 2.03295374e-04 ,-2.02789509e-04 ,-9.99999959e-01]
])

rotation_matrix_2 = np.array([
    [ 8.74262126e-02 ,-9.79531312e-01 , 1.81315115e-01],
    [-9.96170991e-01 ,-8.60061113e-02 , 1.56952231e-02],
    [ 2.20245989e-04, -1.81993031e-01 ,-9.83299812e-01],
])

target_2_optic = np.array([
[ 9.05734241e-01,  3.94176781e-01, -1.55788630e-01],
 [-4.23846006e-01,  8.42332840e-01, -3.32911253e-01],
 [-2.25065278e-17,  3.67559493e-01,  9.30000007e-01],
])

Tool0_2_baselink = np.array([
     [ 4.23846590e-01, -8.42258121e-01,  3.33099503e-01],
 [-9.05733928e-01, -3.94251958e-01,  1.55600107e-01],
 [ 2.69670517e-04, -3.67650079e-01, -9.29964161e-01],
])

now_test = np.array([
    [-4.61919339e-02,  8.69171157e-01, -4.92349255e-01],
 [ 9.98932555e-01,  4.00866897e-02, -2.29520634e-02],
 [-2.12620376e-04, -4.92883901e-01, -8.70095058e-01],
])


axis1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
axis1.rotate(rotation_matrix_1, center=(0, 0, 0))
# axis1.paint_uniform_color([1, 0, 0])

axis2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
axis2.rotate(now_test, center=(0, 0, 0))
# axis2.paint_uniform_color([0, 1, 0])

original_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
# original_axis.paint_uniform_color([1, 0, 0])

o3d.visualization.draw_geometries([axis2, axis1],
                                  window_name="Rotation Matrix Visualization",
                                  width=800,
                                  height=600)




