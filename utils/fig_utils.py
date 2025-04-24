import os
import sys
import json
import pickle
import os.path as osp

import trimesh
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import plotly.graph_objects as go

# TODO: map uv map to colors


def viz_obj_pc_with_grasps(obj_pc, gripper_mesh, RT_grasp_bad, RT_grasp_good):
    """
    Returns a plotly fig with object pc, and fetch gripper mesh at provided RT grasps
    """
    vis_data = []
    vis_data += [
        plot_trimesh_mesh(
            gripper_mesh.copy().apply_transform(RT_grasp_good),
            color="lightgreen",
            opacity=0.8,
        )
    ]
    vis_data += [
        plot_trimesh_mesh(
            gripper_mesh.copy().apply_transform(RT_grasp_bad),
            color="red",
            opacity=0.8,
        )
    ]
    vis_data += [plot_point_cloud(obj_pc)]
    fig = go.Figure(data=vis_data)
    return fig


def plot_point_cloud(pts, color="green", size=3, opacity=0.7):
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker={"color": color, "size": size, "opacity": opacity},
    )


def plot_point_cloud_cmap(pts, color_levels=None, size=6.5):
    colors = cmap_colors(color_levels) if color_levels is not None else "green"
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker={"color": colors, "size": size, "opacity": 1},
    )


def cmap_colors(cmap):
    """
    cmap: numpy array of shape (N,)
    """
    assert cmap.ndim == 1
    num_pts = cmap.shape[0]

    u = (cmap - np.min(cmap)) / (np.max(cmap) - np.min(cmap))

    rgb_colors = np.zeros((num_pts, 3))
    rgb_colors[:, 0] = 1 - u
    rgb_colors[:, 2] = u
    return rgb_colors


def hsv_uv_colors(uv_map, cmap=None):
    """
    uv_map: numpy array of shape (N, 2)
    """
    assert uv_map.ndim == 2
    num_pts = uv_map.shape[0]
    u = uv_map[:, 0]
    v = uv_map[:, -1]
    u = (u - u.min()) / (u.max() - u.min())
    v = (v - v.min()) / (v.max() - v.min())
    # Map scalar1 to hue and scalar2 to brightness
    hue = u
    brightness = v  # can fiddle this if needed
    saturation = cmap if cmap is not None else 1

    # HSV to RGB conversion
    hsv_colors = np.zeros((num_pts, 3))
    hsv_colors[:, 0] = hue  # Hue (0-1)
    hsv_colors[:, 1] = saturation
    hsv_colors[:, 2] = brightness  # Brightness (0-1)

    rgb_colors = np.zeros((num_pts, 4))  # RGBA
    rgb_colors[:, :3] = hsv_to_rgb(hsv_colors)
    rgb_colors[:, -1] = 1  # alpha = 1
    return rgb_colors


def rgbblend_uv_colors(uv_map):
    """
    uv_map: numpy array of shape (N, 2)
    """
    assert uv_map.ndim == 2
    num_pts = uv_map.shape[0]
    u = uv_map[:, 0]
    v = uv_map[:, -1]
    u = (u - u.min()) / (u.max() - u.min())
    v = (v - v.min()) / (v.max() - v.min())
    rgb_colors = np.zeros((num_pts, 3))
    rgb_colors[:, 0] = u  # Hue (0-1)
    rgb_colors[:, 1] = 1 - v  # Brightness (0-1)
    return rgb_colors


def plot_uv_map(pts, color_levels=None, map_func="hsv", size=6.5, opacity=0.9):
    assert map_func in {"hsv", "rgb"}
    if map_func == "hsv":
        uv_mapping_func = hsv_uv_colors
    else:
        uv_mapping_func = rgbblend_uv_colors
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker={
            "color": uv_mapping_func(color_levels),
            "size": size,
            "opacity": opacity,
        },
    )


def cart_to_spherical_coords(pts, radius=1):
    phi = np.arccos(pts[:, 2] / radius)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    theta[theta < 0] += 2 * np.pi
    theta /= 2 * np.pi
    phi /= np.pi
    return np.concatenate(
        [
            np.expand_dims(theta, axis=1),
            np.expand_dims(phi, axis=1),
        ],
        axis=1,
    )


def spherical_to_cart_coords(spherical_pts, radius=1):
    # Extract theta and phi
    theta = spherical_pts[:, 0] * 2 * np.pi  # Undo normalization of theta
    phi = spherical_pts[:, 1] * np.pi  # Undo normalization of phi

    # Convert spherical to cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.stack([x, y, z], axis=1)


def plot_trimesh_mesh(tm_mesh, color="lightblue", opacity=0.3):
    return go.Mesh3d(
        x=tm_mesh.vertices[:, 0],
        y=tm_mesh.vertices[:, 1],
        z=tm_mesh.vertices[:, 2],
        i=tm_mesh.faces[:, 0],
        j=tm_mesh.faces[:, 1],
        k=tm_mesh.faces[:, 2],
        color=color,
        opacity=opacity,
    )


def get_object_mesh(basedir, object_name):
    object_mesh = trimesh.load(
        os.path.join(
            basedir,
            "data/object",
            object_name.split("+")[0],
            object_name.split("+")[1],
            f'{object_name.split("+")[1]}.stl',
        )
    )
    return object_mesh
