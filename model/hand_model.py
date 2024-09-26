import json
import os
import numpy as np
import pickle
import pytorch_kinematics as pk

import torch
import torch.nn
import transforms3d
import trimesh as tm
import urdf_parser_py.urdf as URDF_PARSER
from plotly import graph_objects as go
from pytorch_kinematics.urdf_parser_py.urdf import URDF, Box, Cylinder, Mesh, Sphere

from utils.rot6d_utils import *
from utils.math_utils import *


class GcsHandModel:
    def __init__(
        self,
        robot_name,
        urdf_filename,
        mesh_path,
        urdf_datadir,
        batch_size=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mesh_nsp=128,
        hand_scale=1.0,
    ):
        self.device = device
        self.robot_name = robot_name
        self.batch_size = batch_size
        # prepare model
        self.robot = pk.build_chain_from_urdf(open(urdf_filename).read()).to(
            dtype=torch.float, device=self.device
        )
        self.robot_full = URDF_PARSER.URDF.from_xml_file(urdf_filename)

        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)

        self.surface_points = {}
        self.surface_points_normal = {}
        visual = URDF.from_xml_string(open(urdf_filename).read())
        self.mesh_verts = {}
        self.mesh_faces = {}

        gripper_surface_pts_dict = os.path.join(
            urdf_datadir, "multidex_gripper_surface_pts.pk"
        )

        with open(gripper_surface_pts_dict, "rb") as f:
            _all_surf_pts_dict = pickle.load(f)
        gripper_surface_pts_info = _all_surf_pts_dict[robot_name]

        self.surface_pts_coords = {}
        # Load the precomputed gripper surface points and normals
        link_keys = gripper_surface_pts_info["points"].keys()

        self.gripper_coords_all = (
            torch.from_numpy(gripper_surface_pts_info["coords_all"]).to(device).float()
        )

        for i_link, link_name in enumerate(link_keys):
            pts = gripper_surface_pts_info["points"][link_name]
            pts_normal = gripper_surface_pts_info["normals"][link_name]
            mesh = gripper_surface_pts_info["meshes"][link_name]
            coords = gripper_surface_pts_info["coords_in_link"][link_name]

            # Make into homog coordinates
            pts = np.concatenate([pts, np.ones([len(pts), 1])], axis=-1)
            pts_normal = np.concatenate(
                [pts_normal, np.ones([len(pts_normal), 1])], axis=-1
            )

            self.surface_points[link_name] = (
                torch.from_numpy(pts)
                .to(device)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            self.surface_points_normal[link_name] = (
                torch.from_numpy(pts_normal)
                .to(device)
                .float()
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )

            # No Need to keep a batched version of the coordinates as they are used once
            self.surface_pts_coords[link_name] = (
                torch.from_numpy(coords).to(device).float()
            )

            # visualization mesh
            self.mesh_verts[link_name] = np.array(mesh.vertices)
            self.mesh_faces[link_name] = np.array(mesh.faces)

        # new 2.1
        # Acutally consider both revolute and prismatic joints!
        self.dynamic_joints = []
        for i in range(len(self.robot_full.joints)):
            if self.robot_full.joints[i].joint_type in {"revolute", "prismatic"}:
                self.dynamic_joints.append(self.robot_full.joints[i])
        self.dynamic_joints_q_mid = []
        self.dynamic_joints_q_var = []
        self.dynamic_joints_q_upper = []
        self.dynamic_joints_q_lower = []
        for i in range(len(self.robot.get_joint_parameter_names())):
            for j in range(len(self.dynamic_joints)):
                if (
                    self.dynamic_joints[j].name
                    == self.robot.get_joint_parameter_names()[i]
                ):
                    joint = self.dynamic_joints[j]
            assert joint.name == self.robot.get_joint_parameter_names()[i]
            self.dynamic_joints_q_mid.append(
                (joint.limit.lower + joint.limit.upper) / 2
            )
            self.dynamic_joints_q_var.append(
                ((joint.limit.upper - joint.limit.lower) / 2) ** 2
            )
            self.dynamic_joints_q_lower.append(joint.limit.lower)
            self.dynamic_joints_q_upper.append(joint.limit.upper)

        self.dynamic_joints_q_lower = (
            torch.Tensor(self.dynamic_joints_q_lower)
            .repeat([self.batch_size, 1])
            .to(device)
        )
        self.dynamic_joints_q_upper = (
            torch.Tensor(self.dynamic_joints_q_upper)
            .repeat([self.batch_size, 1])
            .to(device)
        )

        self.current_status = None
        self.scale = hand_scale
        self.palm_normal_dirn = (
            self.get_obj_radius_scale() * self.get_hand_palm_normal()
        )

    def get_obj_radius_scale(self) -> float:
        # scaling factor for object radius
        # obj radius = obj_radius * scale, scale > 1
        gripper = self.robot_name
        if gripper == "Allegro":
            return 1.2
        elif gripper == "Barrett":
            return 1.1
        elif gripper == "fetch_gripper":
            return 1.2
        elif gripper == "franka_panda":
            return 1.2
        elif gripper == "h5_hand":
            return 1.2
        elif gripper == "HumanHand":
            return 1.1
        elif gripper == "jaco_robot":
            return 1.1
        elif gripper == "robotiq_3finger":
            return 1.1
        elif gripper == "sawyer":
            return 1.2
        elif gripper == "shadow_hand":
            return 1.1
        elif gripper == "wsg_50":
            return 1.2
        else:
            return 1

    def get_hand_palm_normal(self):
        gripper = self.robot_name
        if gripper == "Allegro":
            hand_normal = torch.Tensor([[1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "Barrett":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "fetch_gripper":
            hand_normal = torch.Tensor([[1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "franka_panda":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "h5_hand":
            hand_normal = torch.Tensor([[0, 0, -1.0]]).to(self.device).T.float()
        elif gripper == "HumanHand":
            hand_normal = torch.Tensor([[0, -1.0, 0]]).to(self.device).T.float()
        elif gripper == "jaco_robot":
            hand_normal = torch.Tensor([[-1.0, 0, 0]]).to(self.device).T.float()
        elif gripper == "robotiq_3finger":
            hand_normal = torch.Tensor([[0, 1.0, 0]]).to(self.device).T.float()
        elif gripper == "sawyer":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "shadow_hand":
            hand_normal = torch.Tensor([[0, -1.0, 0]]).to(self.device).T.float()
        elif gripper == "wsg_50":
            hand_normal = torch.Tensor([[0, 0, 1.0]]).to(self.device).T.float()
        elif gripper == "barrett":  # gdx
            hand_normal = torch.Tensor([[0.0, 0.0, 1.0]]).to(self.device).T.float()
        elif self.robot_name == "allegro_old":
            hand_normal = torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "shadowhand":
            hand_normal = torch.Tensor([[0.0, -1.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "robotiq_3finger_gdx":
            hand_normal = (
                1.0 * torch.Tensor([[0.0, 0.0, 1.0]]).to(self.device).T.float()
            )
        elif self.robot_name == "ezgripper":
            hand_normal = torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
        elif self.robot_name == "allegro":
            hand_normal = (
                1.0 * torch.Tensor([[1.0, 0.0, 0.0]]).to(self.device).T.float()
            )
        else:
            raise NotImplementedError
        return hand_normal

    def update_kinematics(self, q):
        self.global_translation = q[:, :3]

        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def get_surface_points_prior(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        # if downsample:
        #     surface_points = surface_points[:, torch.randperm(surface_points.shape[1])][:, :778]
        return surface_points * self.scale

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_new(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            if self.robot_name == "robotiq_3finger" and link_name == "gripper_palm":
                continue
            if self.robot_name == "robotiq_3finger_real_robot" and link_name == "palm":
                continue
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_palm(self, q=None):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        if self.robot_name == "allegro":
            palm_list = ["base_link"]
        elif self.robot_name == "robotiq_3finger_real_robot":
            palm_list = ["palm"]
        else:
            raise NotImplementedError
        for link_name in palm_list:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []

        for link_name in self.surface_points:
            # for link_name in parts:
            # get transformation
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix, self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
            surface_normals.append(
                torch.matmul(
                    trans_matrix, self.surface_points_normal[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(
            self.global_rotation, surface_points.transpose(1, 2)
        ).transpose(1, 2) + self.global_translation.unsqueeze(1)
        surface_normals = torch.matmul(
            self.global_rotation, surface_normals.transpose(1, 2)
        ).transpose(1, 2)

        return surface_points, surface_normals

    def get_meshes_from_q(self, q=None, i=0):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = (
                trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            )
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        return data

    def get_plotly_data(self, q=None, i=0, color="lightblue", opacity=1.0):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = (
                trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            )
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones([len(v), 1])], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(
                self.global_rotation[i].detach().cpu().numpy(), transformed_v.T
            ).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(
                go.Mesh3d(
                    x=transformed_v[:, 0],
                    y=transformed_v[:, 1],
                    z=transformed_v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                )
            )
        return data

    def spherical_distance(self, coords1, coords2, radius=1.0):
        """
        Input
        -----
        - coords1: (N, 2) tensor with (theta, phi) values for N points
        - coords2: (M, 2) tensor with (theta, phi) values for M points
        - radius: float, radius of sphere. default is 1 as we only need relative distances!

        Returns
        -------
        - pairwise_dist: (N, M) tensor with distance equal to the haversine distance
        """
        # Convert coordinates from scaled format to angles in radians
        C1 = coords1.clone()
        C2 = coords2.clone()
        C1[:, 0] *= 2 * torch.pi  # Scale theta
        C2[:, 0] *= 2 * torch.pi
        C1[:, 1] *= torch.pi  # Scale phi
        C2[:, 1] *= torch.pi

        # Calculate differences for haversine formula
        diff = C1[:, None] - C2  # Shape (N, M, 2)
        dtheta = diff[:, :, 0]  # Difference in theta
        dphi = diff[:, :, 1]  # Difference in phi

        phi1 = C1[:, 1]  # phi for the first set of points
        phi2 = C2[:, 1]  # phi for the second set of points

        # Haversine formula
        a = (
            torch.sin(dphi / 2) ** 2
            + torch.cos(phi1.unsqueeze(1))
            * torch.cos(phi2)
            * torch.sin(dtheta / 2) ** 2
        )
        c = 2 * torch.arcsin(torch.sqrt(a))

        # Compute the spherical distance
        distances_haversine = radius * c
        return distances_haversine

    def get_gripper_coords(self):
        return self.gripper_coords_all

    def get_correspondence_mask(self, obj_gcs_pred):
        """
        Establishes a mask over the gripper points for the correspondence-based optimization

        Input:
            obj_gcs_pred: torch tensor, shape (N, 2) array with GCS coordinate contact map prediction for each object point


        Returns:
            mask_corr: integer indices mask over the M gripper surface points for the gripper.
                       shape is (N,)
        """
        with torch.no_grad():
            grp_coord = self.gripper_coords_all
            # valid phi values should be greater than 0.3
            obj_mask = obj_gcs_pred[:, 1] > 0.2
            # obj_pts_idxs = torch.where(obj_mask)
            obj_coord = obj_gcs_pred[obj_mask]
            distances = self.spherical_distance(obj_coord, grp_coord)
            sorted_indices = torch.argsort(distances, dim=1)
            corr_grp_pts_indices = sorted_indices[:, 0]  # pick the closest
            return corr_grp_pts_indices, obj_mask

