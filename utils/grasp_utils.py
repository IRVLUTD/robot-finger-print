"""
Utility functions for grasp pose alignment into a common space
Goal: Represent every gripper's pose in a common, aligned way
"""

import os
import numpy as np
import quaternion
import math
import json
from scipy.spatial.transform import Rotation as R

from model.hand_model import GcsHandModel


def rotation_matrix_from_vectors(vec1, vec2):
    """Returns the rotation matrix that aligns vec1 to vec2

    Source:
    https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space

    Normalized versions of vec1_rot = rotmat.dot(vec1) and vec2 should be close then.

    Args:
      vec1: first vector.
      vec2: second vector.

    Returns:
      A 3x3 rotation_matrix/


    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def distance_by_translation_point(p1, p2):
    """
    Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def farthest_points(
    data,
    nclusters,
    dist_func,
    return_center_indexes=False,
    return_distances=False,
    verbose=False,
):
    """
    Performs farthest point sampling on data points.
    Args:
      data: numpy array of the data points.
      nclusters: int, number of clusters.
      dist_dunc: distance function that is used to compare two data points.
      return_center_indexes: bool, If True, returns the indexes of the center of
        clusters.
      return_distances: bool, If True, return distances of each point from centers.

    Returns clusters, [centers, distances]:
      clusters: numpy array containing the cluster index for each element in
        data.
      centers: numpy array containing the integer index of each center.
      distances: numpy array of [npoints] that contains the closest distance of
        each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0], dtype=np.int32), np.arange(
                data.shape[0], dtype=np.int32
            )

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0],), dtype=np.int32) * -1
    distances = np.ones((data.shape[0],), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print("farthest points max distance : {}".format(np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
    If point cloud pc has less points than npoints, it oversamples.
    Otherwise, it downsample the input pc to have npoint points.
    use_farthest_point: indicates whether to use farthest point sampling
    to downsample the points. Farthest point sampling version runs slower.
    """
    if pc.shape[0] >= npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(
                pc, npoints, distance_by_translation_point, return_center_indexes=True
            )
        else:
            center_indexes = np.random.choice(
                range(pc.shape[0]), size=npoints, replace=False
            )
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            center_indexes = np.concatenate([np.arange(pc.shape[0]), index])

    pc = pc[center_indexes, :]
    return pc, center_indexes


def get_quat_pyb(quat_np):
    """
    quat_np: np.quaternion in (w,x,y,z) format
    Return: quaternion list in [x,y,z,w] format (for pybullet)
    """
    return [quat_np.x, quat_np.y, quat_np.z, quat_np.w]


def get_quat_np(quat_pyb):
    """
    quat_pyb: pybullet quaternion in (x,y,z,w) format (as a List)
    Returns: np.quaternion [w,x,y,z] format
    """
    return np.quaternion(quat_pyb[-1], quat_pyb[0], quat_pyb[1], quat_pyb[2])


def get_gripper_common_alignment(gname: str):
    """
    Bring a given gripper's default (urdf) orientation to the common orientation.
    Common orientation: Palm (base link) normal = Z axis, Major axis of Palm = Y axis.

    Input:
        gname: gripper name. Has to be one of: [fetch_gripper, sawyer, franka_panda, wsg_50,
               Barrett, robotiq_3finger, jaco_robot, HumanHand, Allegro, shadow_hand]

    Returns:
        quaternion (w,x,y,z) transformation which when applied on the gripper default urdf, brings it
        into the common alignment orientation described above.
    """
    if gname == "fetch_gripper":
        return quaternion.from_euler_angles([0, -math.pi / 2.0, 0])
    elif gname == "sawyer":
        return np.quaternion(1, 0, 0, 0)  # unit quaternion, i.e NO CHANGE NEEDED
    elif gname == "franka_panda":
        return np.quaternion(1, 0, 0, 0)  # unit quaternion, i.e NO CHANGE NEEDED
    elif gname == "wsg_50":
        return quaternion.from_euler_angles([0, 0, math.pi / 2.0])
    elif gname == "Barrett":
        return np.quaternion(1, 0, 0, 0)  # unit quaternion, i.e NO CHANGE NEEDED
    elif gname == "barrett":  # Gendexgrasp adagrasp barrett
        return np.quaternion(1, 0, 0, 0)  # unit quaternion, i.e NO CHANGE NEEDED
    elif gname == "robotiq_3finger":
        return quaternion.from_euler_angles([0, -math.pi / 2.0, -math.pi / 2.0])
    elif gname == "robotiq_3finger_gdx":
        return np.quaternion(1, 0, 0, 0)
    elif gname == "jaco_robot":
        return quaternion.from_euler_angles([0, math.pi / 2.0, 0])
    elif gname == "HumanHand":
        return quaternion.from_euler_angles([0, math.pi / 2.0, -math.pi / 2.0])
    elif gname == "Allegro" or gname == "allegro":
        return quaternion.from_euler_angles([-math.pi / 2.0, -math.pi / 2.0, 0])
    elif gname == "shadow_hand" or gname == "shadowhand":
        return quaternion.from_euler_angles([math.pi / 2, math.pi / 2, -math.pi / 2])
    elif gname == "h5_hand":
        return quaternion.from_euler_angles([0, math.pi, 0])
    elif gname == "ezgripper":
        return quaternion.from_euler_angles([0, -math.pi / 2.0, 0])
    else:
        print("Invalid gripper name. Returning None!")
        raise NotImplementedError


def get_gripper_palm_position(gname: str, method: str = "isaac_sphere"):
    """
    Call appropriate function for obtaining the aligned palm origin position

    Input:
        gname: gripper name. Has to be one of: [fetch_gripper, sawyer, franka_panda, wsg_50,
               Barrett, robotiq_3finger, jaco_robot, HumanHand, Allegro, shadow_hand]

        method: method to use for palm origin reference. Either "mgg" (used in multigrippergrasp) or "isaac_sphere" used by Felipe

    Returns:
        position vector/offset (x,y,z) for a palm point
    """
    assert method in {"mgg", "isaac_sphere"}
    if method == "mgg":
        return get_gripper_palm_position_mgg(gname)
    elif method == "isaac_sphere":
        return get_gripper_palm_position_isaac_sphere(gname)
    else:
        raise NotImplementedError


def get_gripper_palm_position_mgg(gname: str):
    """
    Used in MultiGripperGrasp
    This function returns the position vector of a point on the gripper's palm surface
    and center of the palm (so think of it like centroid along 2 axes and just on the
    surface for the 3rd axis) -- in the gripper's default urdf frame.

    We are using such a palm point to unify the notion of "position for gripper". The
    position vectors (or offsets) were all computed in the respective gripper URDF's
    base frame (i.e no alignment). Notice the multiplication by -1 here, this means that
    the returned value is indeed the position vector (instead of the offset for the base
    link's position -- this would be true if we didn't have -1)

    Input:
        gname: gripper name. Has to be one of: [fetch_gripper, sawyer, franka_panda, wsg_50,
               Barrett, robotiq_3finger, jaco_robot, HumanHand, Allegro, shadow_hand]

    Returns:
        position vector/offset (x,y,z) for a palm point
    """
    if gname == "fetch_gripper":
        return -1 * np.array([-0.135, 0, 0])
    elif gname == "sawyer":
        return -1 * np.array([0, 0, -0.05])
    elif gname == "franka_panda":
        return -1 * np.array([0, 0, -0.06])
    elif gname == "wsg_50":
        return -1 * np.array([0, 0, -0.072])
    elif gname == "Barrett":
        return -1 * np.array([0, 0, 0])
    elif gname == "robotiq_3finger":
        return -1 * np.array([0, -0.05, 0])
    elif gname == "robotiq_3finger_gdx":
        return -1 * np.array([0, 0, -0.05])
    elif gname == "jaco_robot":
        return -1 * np.array([0.102, 0, 0])
    elif gname == "HumanHand":
        return -1 * np.array([0.1, 0.02, 0])
    elif gname == "Allegro" or gname == "allegro":
        return -1 * np.array([0, 0, 0.03])  # or [-0.01, 0, 0.03]
    elif gname == "shadow_hand":
        return -1 * np.array([0, 0, -0.05])
    elif gname == "shadowhand":
        return -1 * np.array([0, 0, -0.25])
    elif gname == "h5_hand":
        return -1 * np.array([0, 0, 0.045])
    elif gname == "ezgripper":
        return -1 * np.array([-0.082, 0, 0])
    else:
        raise NotImplementedError


def get_gripper_palm_position_isaac_sphere(gname: str):
    """
    This function returns the position vector of a point on the gripper's palm surface
    and center of the palm (so think of it like centroid along 2 axes and just on the
    surface for the 3rd axis) -- in the gripper's default urdf frame.

    We are using such a palm point to unify the notion of "position for gripper". The
    position vectors (or offsets) were all computed in the respective gripper URDF's
    base frame (i.e no alignment). Notice the multiplication by -1 here, this means that
    the returned value is indeed the position vector (instead of the offset for the base
    link's position -- this would be true if we didn't have -1)

    Reference: https://github.com/IRVLUTD/max_sphere/blob/f7afce565f26d7d1c50dfac94bb260f70ed1d3bc/grippers/gripper_isaac_info.json#L12
    See "transfer_reference_pose" key for each gripper for the position.

    Input:
        gname: gripper name. Has to be one of: [fetch_gripper, sawyer, franka_panda, wsg_50,
               Barrett, robotiq_3finger, jaco_robot, HumanHand, Allegro, shadow_hand]

    Returns:
        position vector/offset (x,y,z) for a palm point
    """
    if gname == "fetch_gripper":
        return -1 * np.array([-0.1343, 0, 0])
    elif gname == "sawyer":
        return -1 * np.array([0, 0, -0.0624])
    elif gname == "franka_panda":
        return -1 * np.array([0, 0, -0.066])
    elif gname == "wsg_50":
        return -1 * np.array([0, 0, -0.073])
    elif gname == "Barrett":
        return -1 * np.array([0, 0, 0])
    elif gname == "barrett":
        return -1 * np.array([0, 0, -0.06])
    elif gname == "robotiq_3finger":
        return -1 * np.array([0, -0.05, 0])
    elif gname == "robotiq_3finger_gdx":
        return -1 * np.array([0, 0, -0.092])
    elif gname == "jaco_robot":
        return -1 * np.array([0.1053, 0, 0])
    elif gname == "HumanHand":
        return -1 * np.array([0.16, 0.014, 0.0008])
    elif gname == "Allegro" or gname == "allegro":
        return -1 * np.array([-0.0125, 0, -0.01])  # or [-0.01, 0, 0.03]
    elif gname == "shadow_hand":
        return -1 * np.array([-0.000942, 0.01364, -0.07545])
    elif gname == "shadowhand":
        return -1 * np.array([-0.000942, 0.01364, -0.33])
    elif gname == "h5_hand":
        return -1 * np.array([0, 0, 0.048])
    elif gname == "ezgripper":
        return -1 * np.array([-0.082, 0, 0])
    else:
        raise NotImplementedError


def convert_gripper_to_aligned_pose(
    grasp_pose, source_gripper, quat_mode: str = "xyzw"
):
    """
    Function to bring the grasp pose (position, orientation) 7D vector for a gripper
    into the common alignment space. We transform both the position and orientation
    and return a new "aligned" pose for the gripper

    NOTE: Might be slow since it operates on 1 grasp at a time!

    Input:
        grasp_pose: length 7 numpy array containing (x,y,z) location coordinates
        and the values [x,y,z,w] for the orientation quaternion

        gripper: string containing the gripper's name

    Output:
        Single 7-D Numpy Array for Pose with: [position, orientation_quaternion]
        - position_common: length 3 array containing the aligned position
        - orientation_common: length 3 array containing the aligned orientation (x,y,z,w) for pybullet
    """
    # Get the posn vector (offset) for palm point and convert to np.quaternion (w,x,y,z)

    curr_orn = grasp_pose[3:]
    q_curr_orn = get_quat_np(curr_orn)
    q_align_grp = get_gripper_common_alignment(source_gripper)
    q_orn_common = q_curr_orn * q_align_grp.inverse()
    orientation_common = get_quat_pyb(q_orn_common)

    curr_pos = grasp_pose[:3]
    pv_grp = get_gripper_palm_position(source_gripper)
    q_pv_grp = np.quaternion(
        0, pv_grp[0], pv_grp[1], pv_grp[2]
    )  # Note w = 0 since posn vec
    q_offset_com = q_curr_orn * q_pv_grp * q_curr_orn.inverse()
    offset_common = np.array([q_offset_com.x, q_offset_com.y, q_offset_com.z])
    # offset_common = quaternion.as_vector_part() can also use this
    position_common = curr_pos + offset_common

    # return position_common, orientation_common
    return np.concatenate((position_common, orientation_common))


def convert_aligned_to_gripper_pose(grasp_pose_aligned, target_gripper):
    """
    Function to obtain the base link's (for gripper URDF) position and orientation given
    a grasp pose in the common "aligned" space. Basically, it figures out how to use the
    common aligned pose for a specific (given) gripper.

    NOTE: Might be slow since it operates on 1 grasp at a time!

    Input:
        grasp_pose_aligned: length 7 numpy array containing (x,y,z) location coordinates
        and the values [x,y,z,w] for the orientation quaternion in the "aligned" way i.e
        what the network should ideally predict.

        target_gripper: string containing the target gripper's name -- to which we want to
        apply the aligned grasp pose

    Output:
        Single 7-D Numpy Array for Pose with: [position, orientation_quaternion]
        - position_base: length 3 array for the base link's position
        - orientation_base: length 3 array for the base link's orientation (x,y,z,w)
    """

    orn_com = grasp_pose_aligned[3:]
    q_orn_com = get_quat_np(orn_com)
    q_orn_base = q_orn_com * get_gripper_common_alignment(target_gripper)
    orientation_base = get_quat_pyb(q_orn_base)

    pos_com = grasp_pose_aligned[:3]
    pv_grp = get_gripper_palm_position(target_gripper)
    q_pv_grp = np.quaternion(
        0, pv_grp[0], pv_grp[1], pv_grp[2]
    )  # Note w = 0 since we are representing a position vec using quaternion
    q_offset_base = q_orn_base * q_pv_grp * q_orn_base.inverse()
    position_base = pos_com - np.array(
        [q_offset_base.x, q_offset_base.y, q_offset_base.z]
    )

    # return position_base, orientation_base
    return np.concatenate((position_base, orientation_base))


def convert_7dpose_to_4x4(pose_7d):
    """
    pose_7d: (posn, quat_xyzw) quaternion
    """
    posn = pose_7d[:3]
    quat = get_quat_np(pose_7d[3:])
    transform = np.eye(4)
    transform[:3, :3] = quaternion.as_rotation_matrix(quat)
    transform[:3, 3] = posn
    return transform


def convert_4x4_to_7dpose(pose_matrix):
    """
    Input:
        pose_matrix: 4x4 transformation matrix
    Returns:
        pose_7d: (posn, quat_xyzw) quaternion
    """

    translation = pose_matrix[:3, 3]
    rotation_matrix = pose_matrix[:3, :3]
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # Returns in the order (qx, qy, qz, qw)
    # Combine translation and quaternion into a 7D vector
    pose_7d = np.concatenate([translation, quaternion])
    return pose_7d


def get_handmodel(
    robot,
    batch_size,
    device,
    hand_scale=1.0,
    json_path="urdf_assets_meta.json",
    datadir="./grippers/",
):
    if robot in {
        "barrett",
        "allegro",
        "ezgripper",
        "shadowhand",
        "robotiq_3finger_gdx",
    }:
        # json_path = "data/urdf/urdf_assets_meta.json"
        # datadir = "./dataset/GenDexGrasp/"
        if robot == "robotiq_3finger_gdx":
            robot_name = "robotiq_3finger"
        else:
            robot_name = robot
        urdf_assets_meta = json.load(open(os.path.join(datadir, json_path)))
        urdf_path = os.path.join(datadir, urdf_assets_meta["urdf_path"][robot_name])
        meshes_path = os.path.join(datadir, urdf_assets_meta["meshes_path"][robot_name])
        hand_model = GcsHandModel(
            robot,
            urdf_path,
            meshes_path,
            urdf_datadir=os.path.join(datadir, "data", "urdf"),
            batch_size=batch_size,
            device=device,
            hand_scale=hand_scale,
        )
        return hand_model
    else:
        raise NotImplementedError

