# import the third-party packages
import h5py
import numpy as np
import os
import sys
import open3d as o3d
import torch
from tqdm import tqdm
import torch

# import the local packages
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
from utils.loss_utils import batch_viewpoint_params_to_matrix
from graspnetAPI.utils.utils import generate_views, viewpoint_params_to_matrix

def save_lables_scenes(root, image_id):
    """
    This code is based on https://github.com/maximiliangilles/MetaGraspNet?tab=readme-ov-file
    This function is to convert the meta labels into the graspnet lables, and then save them in the current
    scene foldersw
    **Input:**
    - root: The root number of the scene folder
    - image_id: The id of the image 
    **Output:**
     None, this function will end up saving the ground truth lables
    """
  
    ## read the h5 files in dataset
    h5_file_path = root + str(image_id) + '_scene.hdf5'
    h5_0_file_path = root + '0_scene.hdf5'
    f = h5py.File(str(h5_file_path), 'r')
    f0 = h5py.File(str(h5_0_file_path), 'r')
    grasps_selected = 'non_colliding_grasps'
    
    # read the cameras position, camera_position is for the current scene. camera_position_0 is for the number 0 scene
    camera_position = f['camera']['pose_relative_to_world'][0]
    camera_position_0 = f0['camera']['pose_relative_to_world'][0]
    dset_grasp_width = f[grasps_selected]['paralleljaw']['contact_width']

    #get scores, choose the simulations cores
    dset_score_simulation = f[grasps_selected]['paralleljaw']['score_analytical']
    grasp_rots = []
    points = []
    objects = f0['objects']

    dset_contacts = f[grasps_selected]['paralleljaw']['franka_poses_relative_to_camera']
    object_ids = f[grasps_selected]['paralleljaw']['object_id'][:]
    grasp_widths = np.zeros((len(dset_contacts), 300, 12, 4)) # shape: (N, 300, 12, 4)
    scores = np.zeros((len(dset_contacts), 300, 12, 4)) # shape: (N, 300, 12, 4)
    contact_poses = list(dset_contacts)
    
    
    #generate 300 views
    template_views = generate_views(300)
    # generating 12 angles
    num_angles = 12
    angles = np.array([np.pi / num_angles * i for i in range(num_angles)], dtype=np.float32)
    # generating 4 depths
    num_depths = 4
    depths = np.array([0.01 * i for i in range(1, num_depths+1)], dtype=np.float32)
    
    number = -1
    print('contact_poses', len(contact_poses))
    
    
    for idx, contact_transform in enumerate(contact_poses):
        gripper = o3d.io.read_triangle_mesh("/home/zhy/Grasp_pointcloud/new_structure/MetaGraspNet/utils/Meshes/parallel_gripper.ply")
        gripper.scale(scale=0.01, center=np.array([0,0,0]))
        vertices = np.asarray(gripper.vertices)
        max_distance = 0
        center = gripper.get_center()
        for vertex in vertices:
            distance = abs(vertex[0] - center[0])  # Distance from center in x-axis

            if distance > max_distance:
                max_distance = distance
        # Here width is calculated using the gripper mesh's width, so now the width is the same
        width = 2 * max_distance
        # Here point is divided by 100, so we need to times 100, and then the result needs to be divided by 100
            # to keep it in the original scale
        contact_transform[0,3] /=100
        contact_transform[1,3] /=100
        contact_transform[2,3] /=100
        grasp_rot = contact_transform[:3, :3]

        # The gripper needs to transform by the contact_transform
        gripper.transform(contact_transform)
        obb = gripper.get_oriented_bounding_box()
        point = obb.get_center()
        
        
        object_id = object_ids[idx]

        if int(object_id) >= len(objects['poses_relative_to_camera']):
            depth = 0.04
        else:
            object_position = objects['poses_relative_to_camera'][int(object_id)][:]
            object_position_point = object_position[:3, 3]/100
            object_position_point = (object_position_point + camera_position_0[:3, 3] - camera_position[:3, 3])
            position_difference = (point - object_position_point)
            depth = abs(position_difference[2])


        # The combined rotation is form transform the gripper, the original position fo meta gripper and graspnet gripper is different
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
        rotation_z = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, -np.pi/2))
        combined_rotation = np.dot(grasp_rot, np.dot(rotation_z, rotation))

        if not any(np.array_equal(point, p) for p in points):
            number = number + 1
            points.append(point)
            grasp_rots.append(combined_rotation)
                    
        point_number = find_point_index(points, point)
        grasp_rot = combined_rotation.copy()
        towards, angle = matrix_to_viewpoint(grasp_rot)

        #Fix the angle problem
        if angle > np.pi:
            angle = angle - np.pi
        if angle < 0:
            angle = angle +np.pi

        # Finding the closest angle in 12 angles, the closest depth in 4 depths
        angles_index = find_closest_index(angle, angles)
        depths_index = find_closest_index(depth, depths)
        vector_similarity, view_index = compare_vectors_from_list(template_views, -towards)
        if vector_similarity > 0:
            # grasp_rot_converted = viewpoint_params_to_matrix(-template_views[view_index], angles[angles_index])
            # grasp_rots.append(grasp_rot_converted)
            score = dset_score_simulation[idx]
            width = dset_grasp_width[idx]/100
            if scores[point_number, view_index, angles_index, depths_index] != 0:
                print('already have points!')
            scores[point_number, view_index, angles_index, depths_index] = score # shape: (N, 300, 12, 4)
            grasp_widths[point_number, view_index, angles_index, depths_index] = width # shape: (N, 300, 12, 4)
        else:
            print('did not find any')

           
    print('number', number)
    scores = scores[:number + 1]
    grasp_widths = grasp_widths[:number + 1]
    


    #save the files
    np.savez(os.path.join(root, '{}_labels.npz'.format(str(image_id).zfill(3))),
                 points=torch.tensor(points), scores=torch.tensor(scores), 
                 width=torch.tensor(grasp_widths), rot = torch.tensor(grasp_rots))

    
def compare_vectors_from_list(view_points, toward_vector):
    """
    Find the closest view of toward_vector from the view_points list
    
    **Input:**
    - view_points: List of views 
    - toward_vector: a toward vector
    **Output:**
    - max_dot_product: The maximum dot product between toward_vector and the view_points list,
    if it is larger than 0.9, then there a view found in the view_point list, if the value is smaller
    than 0, then there is no closet view found
    - max_index: The index of the closest view found
    """
    max_dot_product = 0
    max_index = 0
    for i, view_point in enumerate(view_points):
        # Calculate vector from viewpoint to grasp point
        vg = np.array(view_point) 
        
        # Normalize both vectors
        vg_normalized = vg / abs(np.linalg.norm(vg))
        t_normalized = np.array(toward_vector) / abs(np.linalg.norm(toward_vector))
        
        # Calculate dot product
        dot_product = np.dot(vg_normalized, t_normalized)
        
        # Interpret result
        if dot_product > 0.9:  
            if dot_product >= max_dot_product:
                # This should find the closest views
                max_dot_product = dot_product
                max_index = i
    return max_dot_product, max_index

def find_point_index(points, point):
    """
    Find the exact point in the points list
    
    **Input:**
    - points: List of points 
    - point: The point that needs to be found in the list
    **Output:**
    - index: The index of the point found
    """
    # Ensure points and point are numpy arrays
    points = np.asarray(points)
    point = np.asarray(point)
    
    # Find rows in points that match point
    matches = np.all(points == point, axis=1)
    
    # Get the indices of matches
    indices = np.where(matches)[0]
    
    if len(indices) == 0:
        raise ValueError("Point not found in the array")
    
    # Return the first matching index
    index = indices[0]
    return index

def find_closest_index(target, float_list):
    # Use enumerate to get both index and value
    index, closest_value = min(enumerate(float_list), key=lambda x: abs(x[1] - target))
    return index


def matrix_to_viewpoint(rotation_matrix):
    """
    Convert a 4x4 transformation matrix to viewpoint parameters and verify reversibility.
    
    **Input:**
    - rotation_matrix: numpy array of shape (4, 4) or (3, 3) representing a transformation matrix.
    **Output:**
    - towards: numpy array of shape (3,) representing the direction vector.
    - angle: float representing the in-plane rotation angle.
    """
    # Ensure we're working with a 3x3 rotation matrix
    original_matrix = rotation_matrix.copy()
    if rotation_matrix.shape == (4, 4):
        rotation_matrix = rotation_matrix[:3, :3]
    
    towards = rotation_matrix[:, 0]
    original_z = rotation_matrix[:, 2]
    
    # Reconstruct axis_y
    axis_y = np.array([-towards[1], towards[0], 0])
    if np.linalg.norm(axis_y) == 0:
        axis_y = np.array([0, 1, 0])
    axis_y = axis_y / np.linalg.norm(axis_y)
    
    # Reconstruct the original coordinate system
    axis_x = towards / np.linalg.norm(towards)
    axis_z = np.cross(axis_x, axis_y)
    
    # Check if the reconstructed z-axis is in the opposite direction of the original z-axis
    if np.dot(axis_z, original_z) < 0:
        axis_z = -axis_z
        axis_y = -axis_y
    
    R2 = np.column_stack((axis_x, axis_y, axis_z))
    
    # Extract R1 from the original matrix
    R1 = np.dot(R2.T, rotation_matrix)
    
    # Extract the angle from R1
    angle = np.arctan2(R1[2, 1], R1[1, 1])
    
    # Verify reversibility
    reconstructed_matrix = batch_viewpoint_params_to_matrix(torch.tensor([towards], dtype=torch.float32), torch.tensor([angle], dtype=torch.float32))[0]
    if np.round(original_matrix[0, 2]) * np.round(reconstructed_matrix[0,2]) < 0:
        angle = angle+np.pi
    return towards, angle


if __name__ == '__main__':
    root = '/media/zhy/Data2TB1/MetaGraspNet/mnt/data1/data_ifl'
    for folder in tqdm(range(2131, 2210)):
        print('started scene:', folder)
        folder = 'scene' + str(folder)
        for image_id in range(10):
            # print(image_id)
            scene_root = root + '/' + folder + '/'
            save_lables_scenes(scene_root, image_id)
   


    

