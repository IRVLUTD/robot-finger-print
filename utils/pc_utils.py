import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.distance import cdist


def compute_contact_map(gripper_pts, obj_pts, sharp_factor):
    pairwise_distances = cdist(obj_pts, gripper_pts)
    distances = np.min(pairwise_distances, axis=1)

    # contact_m = 1 - 2 * (sigmoid(100 * distances))
    contact_m = 2 / (1 + np.exp(sharp_factor * distances))
    return contact_m


def compute_contact_map_aligned(gripper_pts, obj_pts, obj_normals, sharp_factor):
    pairwise_distances = cdist(obj_pts, gripper_pts)

    pairwise_differences = obj_pts[:, np.newaxis] - gripper_pts
    pairwise_alignments = np.sum(
        pairwise_differences * obj_normals[:, np.newaxis, :], axis=-1
    )
    pairwise_alignments = np.exp(1 - pairwise_alignments)

    distances = np.min(pairwise_distances * pairwise_alignments, axis=1)
    contact_m = 2 / (1 + np.exp(sharp_factor * distances))
    return contact_m


def apply_extrinsics(points, RT_camera):
    """
    Transform the point cloud using the camera extrinsics.

    Args:
        points: (N,3) numpy array.
        RT_camera: (4,4) tf (rotation and translation) for camera

    Returns:
        Transformed Open3D PointCloud object.
    """
    rotation = RT_camera[:3, :3]
    translation = RT_camera[:3, 3]
    transformed_points = (rotation @ points.T).T + translation.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    return pcd


def estimate_normals_with_open3d(point_cloud, camera_position=None):
    """
    Estimate normals of a point cloud and orient them.

    Args:
        point_cloud: Open3D PointCloud object.
        camera_position: Optional 3x1 numpy array of camera position for normal orientation.

    Returns:
        Point cloud with estimated normals.
    """
    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Orient normals towards the camera or in a consistent direction
    if camera_position is not None:
        point_cloud.orient_normals_towards_camera_location(camera_position)
    else:
        point_cloud.orient_normals_consistent_tangent_plane(k=30)

    return point_cloud


def transform_to_camera_frame(point_cloud, RT_camera):
    """
    Transform both points and normals from the world frame to the camera frame.

    Args:
        point_cloud: Open3D PointCloud object with estimated normals.
        rotation: 3x3 numpy array representing the rotation matrix of the camera extrinsics.
        translation: 3x1 numpy array representing the translation vector of the camera extrinsics.

    Returns:
        Open3D PointCloud object with points and normals in the camera frame.
    """
    # Get points and normals from the point cloud
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    rotation = RT_camera[:3, :3]
    translation = RT_camera[:3, 3]
    translation = translation.reshape(1, 3)

    # Compute inverse rotation matrix
    rotation_inverse = rotation.T  # Inverse of rotation (R^-1)

    # Transform points to the camera frame
    transformed_points = (rotation_inverse @ (points - translation).T).T

    # Transform normals to the camera frame
    transformed_normals = (rotation_inverse @ normals.T).T

    # Update the point cloud with transformed points and normals
    tf_pcd = o3d.geometry.PointCloud()
    tf_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    tf_pcd.normals = o3d.utility.Vector3dVector(transformed_normals)
    return tf_pcd


def filter_outliers(traj, threshold=0.2):
    """
    Filters outliers from a list of 4x4 homogeneous matrices.
    Args:
        traj (list of np.ndarray): The trajectory as a list of 4x4 matrices.
        threshold (float): Maximum allowed distance between consecutive poses.
    Returns:
        list of np.ndarray: Filtered trajectory.
    """
    filtered_traj = [traj[0]]

    for i in range(1, len(traj)):
        prev_pos = traj[i - 1][:3, 3]
        curr_pos = traj[i][:3, 3]

        dist = np.linalg.norm(curr_pos - prev_pos)

        if dist <= threshold:
            filtered_traj.append(traj[i])

    return filtered_traj


def load_depth_img(img_path):
    """
    Loads a depth image corresponding to the given image path.

    It reads the depth image, normalizes it by dividing by 1000 (to convert the depth
    values from millimeters to meters), and returns the depth data as a NumPy array.

    Source: https://github.com/IRVLUTD/hamer-depth/commit/070886168e469ab1645612a2c3b8c6473aab1aef#diff-6bacd8700314864adb2bf1d56bb841dab8e0ac87d88c8303caa83b545d0b4b9dR116

    Args:
        img_path (str): Path to the depth image file.

    Returns:
        np.ndarray: Normalized depth image as a NumPy array.

    Raises:
        FileNotFoundError: If the depth image file does not exist.
        ValueError: If the depth image cannot be loaded or is invalid.
    """
    try:
        # Replace 'rgb' with 'depth' and change the extension to '.png'
        depth_path = str(img_path).replace("rgb", "depth").replace("jpg", "png")

        # Read the depth image
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"Failed to load depth image from {depth_path}")

        # Convert depth to float32 and normalize
        depth = depth.astype(np.float32) / 1000.0

        return depth

    except FileNotFoundError as e:
        print(f"Depth image file not found: {e}")
        raise

    except ValueError as e:
        print(f"Error loading depth image: {e}")
        raise

    except Exception as e:
        print(f"An unexpected error occurred while loading the depth image: {e}")
        raise


def compute_xyz(depth_img, fx, fy, px, py):
    height, width = depth_img.shape
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def backproject_camera(im_depth, K, target_mask=None, threshold=5):
    Kinv = np.linalg.inv(K)

    width = im_depth.shape[1]
    height = im_depth.shape[0]
    depth = im_depth.astype(np.float32, copy=True).flatten()
    if target_mask is not None:
        mask = (depth > 0) & (depth < threshold) & (target_mask.flatten() > 0)
    else:
        mask = (depth > 0) & (depth < threshold)

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)  # each pixel

    # backprojection
    R = Kinv.dot(x2d.transpose())
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
    return X[:, mask].T
