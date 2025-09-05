import numpy as np
import pykitti
import open3d as o3d

Odo_Dir = 'KITTI_SAMPLE/ODOMETRY'
RAW_Dir = 'KITTI_SAMPLE/RAW'
date = '2011_09_26'
drive = '0009'

# Load the data
data = pykitti.raw(RAW_Dir, date, drive)

all_points = []
all_colors = []

# Loop through each frame, process and add points
for i in range(51):

    # Load LiDAR points
    velo = data.get_velo(i)
    # Filter points: remove very close or very far points
    velo_data_clipped = (velo[:, 0] > 2) & (velo[:, 0] < 50)
    velo = velo[velo_data_clipped]
    # filter out reflective points
    points = velo[:, :3]

    # Transform LiDAR points to world frame
    pose = data.oxts[i].T_w_imu  # 4x4 transformation matrix, rotation and translation
    points_h = np.hstack((points, np.ones((points.shape[0], 1)))) # adds 1 (x, y, z, 1) n x 4
    points_world = (pose @ points_h.T).T[:, :3] # matrix transformation but points made into vertical form to allow for multiplication

    # Same size array as point world, fill with Default gray color
    colors = np.ones_like(points_world) * 0.5

    # Color from camera image
    image = np.array(data.get_cam2(i), dtype=np.uint8) # Creates an array of pixels to access later
    calcam = data.calib.T_cam2_velo # matrix with extrinsic parameters to go from lidar to camera coordinates
    K = np.array(data.calib.K_cam2) # internal parameter matrix

    # Transform points to camera frame
    points_cam = (calcam @ points_h.T).T[:, :3]

    # Only keep points in front of camera
    mask_front = points_cam[:, 2] > 0
    points_cam = points_cam[mask_front]
    points_world = points_world[mask_front]
    colors = colors[mask_front]

    # Project points to image plane
    proj = (K @ points_cam.T).T
    u = (proj[:, 0] / proj[:, 2]).astype(int)
    v = (proj[:, 1] / proj[:, 2]).astype(int)

    # Keep points inside image bounds
    mask_img = (u >= 0) & (u < image.shape[1]) & (v >= 0) & (v < image.shape[0])
    u, v = u[mask_img], v[mask_img]
    points_world = points_world[mask_img]
    colors = colors[mask_img]

    # Assign colors from image
    colors = image[v, u, :] / 255.0

    # Collect points
    all_points.append(points_world)
    all_colors.append(colors)

# Merge all frames
all_points = np.vstack(all_points)
all_colors = np.vstack(all_colors)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

# smoother visualization
pcd = pcd.voxel_down_sample(voxel_size=0.01)

# remove statistical outliers
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

# Visualize
o3d.visualization.draw_geometries([pcd])
