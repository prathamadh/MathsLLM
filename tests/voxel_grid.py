import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def create_voxel_representation(rgb_image_path, depth_image_path):
    # Load images
    rgb_image = cv2.imread(rgb_image_path)
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)

    # Convert to numpy arrays
    rgb_array = np.array(rgb_image)
    depth_array = np.array(depth_image)

    # Get width and height from shape
    height, width, _ = rgb_array.shape
    max_depth = np.max(depth_array)

    # Define maximum depth limit for the voxel grid
    max_depth_limit = int(min(max_depth, 1000))  # Limit max depth for safety

    # Create voxel grid with the correct shape
    voxel_grid = np.zeros((width, height, max_depth_limit + 1, 3))  # RGB in 4th dimension

    # Populate voxel grid
    for y in range(height):
        for x in range(width):
            depth = int(depth_array[y, x])
            if 0 < depth <= max_depth_limit:  # Ignore invalid depth readings and check max depth
                voxel_grid[x, y, depth] = rgb_array[y, x] / 255.0  # Assign normalized RGB values

    return voxel_grid

def visualize_voxel_grid(voxel_grid):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=45)
    # Get occupied voxels and their colors
    occupied_voxels = np.argwhere(np.any(voxel_grid != 0, axis=-1))
    colors = voxel_grid[occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2]]

    # Plot voxels
    ax.scatter(occupied_voxels[:, 0], occupied_voxels[:, 1], occupied_voxels[:, 2],
               c=colors, s=1)

    plt.title('Voxel Representation')
    plt.show()

# Usage
rgb_image_path = 'D:\\desktopbackup\\deepseekmath\\backup\\IntelliMath-Solver\\tests\\Grand-Canyon-Depth-Map.png'
depth_image_path = 'D:\\desktopbackup\\deepseekmath\\backup\\IntelliMath-Solver\\tests\\depth map.png'

voxel_grid = create_voxel_representation(rgb_image_path, depth_image_path)
visualize_voxel_grid(voxel_grid)