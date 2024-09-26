import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np

import torch
import torch.nn.functional as F
from utils.grasp_utils import (
    get_handmodel,
    convert_aligned_to_gripper_pose,
    convert_7dpose_to_4x4,
    convert_4x4_to_7dpose,
    rotation_matrix_from_vectors,
)


class CMapAdam:
    def __init__(
        self,
        robot_name,
        contact_map_goal=None,
        num_particles=32,
        init_rand_scale=0.5,
        learning_rate=5e-3,
        running_name=None,
        energy_func_name="align_dist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose_energy=False,
    ):
        self.running_name = running_name
        self.device = device
        self.robot_name = robot_name
        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.verbose_energy = verbose_energy

        self.global_step = None
        self.contact_map_goal = None
        self.q_current = None
        self.energy = None

        self.compute_energy = None
        self.object_radius = None
        self.contact_value_goal = None
        self.object_point_cloud = None
        self.object_normal_cloud = None

        self.q_global = None
        self.q_local = None
        self.optimizer = None

        self.handmodel = get_handmodel(
            robot_name,
            num_particles,
            device,
            hand_scale=1.0,
            json_path="urdf_assets_meta.json",
            datadir="./grippers",
        )
        self.q_joint_lower = self.handmodel.dynamic_joints_q_lower.detach()
        self.q_joint_upper = self.handmodel.dynamic_joints_q_upper.detach()

        if contact_map_goal is not None:
            self.reset(
                contact_map_goal=contact_map_goal,
                running_name=running_name,
                energy_func_name=energy_func_name,
            )

    def reset(self, contact_map_goal, running_name, energy_func_name):
        self.handmodel = get_handmodel(
            self.robot_name,
            self.num_particles,
            self.device,
            hand_scale=1.0,
            json_path="urdf_assets_meta.json",
            datadir="./grippers/",
        )
        energy_func_map = {
            "euclidean_dist": self.compute_energy_euclidean_dist,
            "align_dist": self.compute_energy_align_dist,
        }
        self.compute_energy = energy_func_map[energy_func_name]

        self.running_name = running_name
        self.is_pruned = False
        self.best_index = None
        self.global_step = 0
        self.distance_init = 1.0
        self.contact_map_goal = contact_map_goal.to(self.device)
        self.object_point_cloud = contact_map_goal[:, :3].to(self.device)
        self.object_normal_cloud = contact_map_goal[:, 3:6].to(self.device)
        self.contact_value_goal = contact_map_goal[:, 6].to(self.device)
        self.object_radius = torch.max(torch.norm(self.object_point_cloud, dim=1, p=2))

        # initialize the opt for grasp = (posn, rotn, dof joints)
        self.q_current = torch.zeros(
            self.num_particles,
            3 + 6 + len(self.handmodel.dynamic_joints),
            device=self.device,
        )
        random_rot = torch.tensor(
            R.random(self.num_particles).as_matrix(), device=self.device
        ).float()

        # NOTE: For a rotmat R, R.T.reshape(9)[:6] will give the ortho6d elements in order
        # I.e x1,x2,x3,y1,y2,y3
        # GenDexGrasp original Code here had a Bug! Although it didnt matter as they were random rotations
        # self.q_current[:, 3:9] = random_rot.reshape(self.num_particles, 9)[:, :6]
        self.q_current[:, 3:9] = random_rot.transpose(1, 2).reshape(
            self.num_particles, 9
        )[:, :6]
        # # TODO: for debug
        # self.handmodel.update_kinematics(q=self.q_current)
        hand_center_position = torch.mean(
            self.handmodel.get_surface_points(q=self.q_current), dim=1
        )
        if (
            self.robot_name == "allegro"
            or self.robot_name == "robotiq_3finger_real_robot"
        ):
            hand_center_position = torch.mean(
                self.handmodel.get_surface_points_palm(q=self.q_current), dim=1
            )

        # HAND NORMAL VECTOR
        hand_normal = self.handmodel.palm_normal_dirn
        hand_normal *= self.distance_init
        hand_normal = torch.einsum(
            "bmn,nk->bmk", random_rot.transpose(2, 1), hand_normal
        ).squeeze(2)

        self.q_current[:, :3] = -hand_center_position
        self.q_current[:, :3] -= hand_normal * self.object_radius
        # Set the dof values to be initialized between (lower, lower + range * rand_0_1 * scale)
        self.q_current[:, 9:] = (
            self.init_random_scale
            * torch.rand_like(self.q_current[:, 9:])
            * (self.q_joint_upper - self.q_joint_lower)
            + self.q_joint_lower
        )
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

    def compute_energy_euclidean_dist(self):
        hand_surface_points_ = self.handmodel.get_surface_points()
        hand_surface_points = hand_surface_points_.clone()

        # compute contact value with align dist
        npts_object = self.object_point_cloud.size()[0]
        npts_hand = hand_surface_points.size()[1]
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )
        hand_surface_points = hand_surface_points.reshape(
            self.num_particles, 1, npts_hand, 3
        )
        batch_object_point_cloud = batch_object_point_cloud.repeat(
            1, npts_hand, 1, 1
        ).transpose(1, 2)
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)

        contact_dist = object_hand_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (torch.sigmoid(100 * contact_dist) - 0.5)
        energy_contact = torch.abs(
            contact_value_current - self.contact_value_goal.reshape(1, -1)
        ).mean(dim=1)

        # compute penetration
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )
        hand_surface_points = hand_surface_points_.reshape(
            self.num_particles, 1, npts_hand, 3
        )
        hand_surface_points = hand_surface_points.repeat(
            1, npts_object, 1, 1
        ).transpose(1, 2)
        batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1)
        hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack(
            [self.object_point_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_normal = torch.stack(
            [self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_signs = (
            (hand_object_points - hand_surface_points_) * hand_object_normal
        ).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)
        energy = energy_contact + 100 * energy_penetration
        self.energy = energy
        # TODO: add a normalized energy
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
            self.q_joint_lower - self.q_current[:, 9:]
        )
        self.energy = energy + z_norm.sum(dim=1)
        if self.verbose_energy:
            return energy, energy_penetration, z_norm
        else:
            return energy

    def compute_energy_align_dist(self):
        # hand_surface_points_ = self.handmodel.get_surface_points()
        hand_surface_points_ = self.handmodel.get_surface_points_new()
        hand_surface_points = hand_surface_points_.clone()
        # compute contact value with align dist
        npts_object = self.object_point_cloud.size()[0]
        npts_hand = hand_surface_points.size()[1]
        with torch.no_grad():
            batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
                self.num_particles, 1, 1
            )
            batch_object_point_cloud = batch_object_point_cloud.view(
                self.num_particles, 1, npts_object, 3
            )
            batch_object_point_cloud = batch_object_point_cloud.repeat(
                1, npts_hand, 1, 1
            ).transpose(1, 2)
        hand_surface_points = hand_surface_points.view(
            self.num_particles, 1, npts_hand, 3
        )
        hand_surface_points = hand_surface_points.repeat(1, npts_object, 1, 1)

        with torch.no_grad():
            batch_object_normal_cloud = self.object_normal_cloud.unsqueeze(0).repeat(
                self.num_particles, 1, 1
            )
            batch_object_normal_cloud = batch_object_normal_cloud.view(
                self.num_particles, 1, npts_object, 3
            )
            batch_object_normal_cloud = batch_object_normal_cloud.repeat(
                1, npts_hand, 1, 1
            ).transpose(1, 2)
        object_hand_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        object_hand_align = (
            (hand_surface_points - batch_object_point_cloud) * batch_object_normal_cloud
        ).sum(dim=3)
        object_hand_align /= object_hand_dist + 1e-5

        object_hand_align_dist = object_hand_dist * torch.exp(
            2 * (1 - object_hand_align)
        )
        # TODO: add a mask of back points
        # object_hand_align_dist = torch.where(object_hand_align > 0, object_hand_align_dist,
        #                                      torch.ones_like(object_hand_align_dist))

        contact_dist = torch.sqrt(object_hand_align_dist.min(dim=2)[0])
        # contact_dist = object_hand_align_dist.min(dim=2)[0]
        contact_value_current = 1 - 2 * (torch.sigmoid(10 * contact_dist) - 0.5)
        energy_contact = torch.abs(
            contact_value_current - self.contact_value_goal.view(1, -1)
        ).mean(dim=1)

        # compute penetration
        with torch.no_grad():
            batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
                self.num_particles, 1, 1
            )
            batch_object_point_cloud = batch_object_point_cloud.view(
                self.num_particles, 1, npts_object, 3
            )
            batch_object_point_cloud = batch_object_point_cloud.repeat(
                1, npts_hand, 1, 1
            )

        hand_surface_points = hand_surface_points_.view(
            self.num_particles, 1, npts_hand, 3
        )
        hand_surface_points = hand_surface_points.repeat(
            1, npts_object, 1, 1
        ).transpose(1, 2)
        hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack(
            [self.object_point_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_normal = torch.stack(
            [self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0
        )
        # torch.gather()
        hand_object_signs = (
            (hand_object_points - hand_surface_points_) * hand_object_normal
        ).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)

        energy = energy_contact + 100 * energy_penetration
        # energy = energy_contact
        # TODO: add a normalized energy
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
            self.q_joint_lower - self.q_current[:, 9:]
        )

        self.energy = energy + z_norm.sum(dim=1)

        if self.verbose_energy:
            return energy, energy_penetration, z_norm
        else:
            return energy

    def step(self):
        self.optimizer.zero_grad()
        self.handmodel.update_kinematics(q=self.q_current)
        energy = self.compute_energy()
        # if self.is_pruned:
        #     energy = energy[self.best_index]
        energy.mean().backward()
        self.optimizer.step()
        self.global_step += 1

    def do_pruning(self):
        raise NotImplementedError
        self.best_index = self.energy.min(dim=0)[1].item()
        # todo: restart optimizer?
        self.handmodel = get_handmodel(self.robot_name, 1, self.device, hand_scale=1.0)
        self.q_current = self.q_current[[self.best_index], :].detach()
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate / 5)
        self.is_pruned = True

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color="pink", opacity=0.7):
        # self.handmodel.update_kinematics(q=self.q_current)
        return self.handmodel.get_plotly_data(
            q=self.q_current, i=index, color=color, opacity=opacity
        )


class GcsHandOpt:

    def __init__(
        self,
        robot_name,
        contact_map_goal=None,
        num_particles=32,
        init_rand_scale=0.5,
        learning_rate=5e-3,
        running_name=None,
        energy_func_name="euclidean_dist",
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose_energy=False,
        penetration_energy_weight=60,
    ):
        self.running_name = running_name
        self.device = device
        self.robot_name = robot_name
        self.num_particles = num_particles
        self.init_random_scale = init_rand_scale
        self.learning_rate = learning_rate

        self.verbose_energy = verbose_energy

        self.global_step = None
        self.contact_map_goal = None
        self.q_current = None
        self.energy = None

        self.compute_energy = None
        self.object_radius = None
        self.contact_value_goal = None
        self.object_point_cloud = None
        self.object_normal_cloud = None
        self.penetration_energy_weight = penetration_energy_weight

        self.grp_corr_idxs = None
        # self.q_local = None
        self.optimizer = None

        # For GenDexGrasp Grippers
        if robot_name in {
            "barrett",
            "allegro",
            "ezgripper",
            "shadowhand",
            "robotiq_3finger_gdx",
        }:
            self._gripper_json_path = "data/urdf/urdf_assets_meta.json"
            self._gripper_datadir = os.path.expanduser("~/Datasets/GenDexGrasp")
        else:
            raise NotImplementedError

        self.handmodel = get_handmodel(
            robot_name,
            num_particles,
            device,
            hand_scale=1.0,
            json_path=self._gripper_json_path,
            datadir=self._gripper_datadir,
        )
        self.q_joint_lower = self.handmodel.dynamic_joints_q_lower.detach()
        self.q_joint_upper = self.handmodel.dynamic_joints_q_upper.detach()

        if contact_map_goal is not None:
            self.reset(
                contact_map_goal=contact_map_goal,
                running_name=running_name,
                energy_func_name=energy_func_name,
            )

    def reset(
        self,
        contact_map_goal,
        running_name,
        energy_func_name="euclidean_dist",
        ub_theta=0.3,
        lb_phi=0.6,
    ):
        """
        NOTE: ub_theta, lb_phi (0.1, 0.8) might be too strict of a threshold for palm coords
        """

        self.handmodel = get_handmodel(
            self.robot_name,
            self.num_particles,
            self.device,
            hand_scale=1.0,
            json_path=self._gripper_json_path,
            datadir=self._gripper_datadir,
        )
        if energy_func_name not in {"euclidean_dist"}:
            raise NotImplementedError

        energy_func_map = {
            "euclidean_dist": self.compute_energy_euclidean_dist,
            "align_dist": None,  # self.compute_energy_align_dist,
        }

        self.compute_energy = energy_func_map[energy_func_name]

        self.running_name = running_name
        self.is_pruned = False
        self.best_index = None
        self.global_step = 0
        self.distance_init = 1.0
        self.contact_map_goal = contact_map_goal.to(self.device)
        self.object_point_cloud = contact_map_goal[:, :3].to(self.device)
        self.object_normal_cloud = contact_map_goal[:, 3:6].to(self.device)

        # Here contact value refer to the predicted (gcs) coordinate contact map predicted on object surface
        self.contact_value_goal = contact_map_goal[:, 6:].to(self.device)
        self.object_radius = torch.max(torch.norm(self.object_point_cloud, dim=1, p=2))

        # Get correspondence between predicted obj coords and gripper coords
        # Use this for the closest energy distance computation and penetration
        self.grp_corr_idxs, self.obj_corr_mask = self.handmodel.get_correspondence_mask(
            self.contact_value_goal
        )

        # initialize the opt for grasp = (posn, rotn, dof joints)
        self.q_current = torch.zeros(
            self.num_particles,
            3 + 6 + len(self.handmodel.dynamic_joints),
            device=self.device,
        )

        # TODO: Take in the thresholds as hyper-params. (ideal coord is (0, 1))
        with torch.no_grad():
            palm_center_mask = (self.contact_value_goal[:, 0] <= ub_theta) & (
                self.contact_value_goal[:, 1] >= lb_phi
            )
            # TODO: Ideally implement in a loop and slowly adjust bounds
            if not palm_center_mask.any():
                # our bound was too strict!
                print("Adjusting bounds....")
                palm_center_mask = (
                    self.contact_value_goal[:, 0] <= (ub_theta + 0.1)
                ) & (self.contact_value_goal[:, 1] >= (lb_phi - 0.1))
            if not palm_center_mask.any():
                # our bound was too strict!
                print("Adjusting bounds....")
                palm_center_mask = (
                    self.contact_value_goal[:, 0] <= (ub_theta + 0.15)
                ) & (self.contact_value_goal[:, 1] >= (lb_phi - 0.15))

            avg_palm_pts = torch.mean(self.object_point_cloud[palm_center_mask], dim=0)
            avg_palm_pts_norm = torch.mean(
                self.object_normal_cloud[palm_center_mask], dim=0
            )
            avg_norm_dir = avg_palm_pts_norm / torch.norm(
                avg_palm_pts_norm, dim=0, keepdim=True
            )
            target_pos = avg_palm_pts + avg_norm_dir * 0.1

        # In the aligned pose space, all grippers have palm normal as +Z
        og_hand_normal = np.array([0, 0, 1])
        target_hand_normal = -1 * avg_norm_dir.clone().detach().cpu().numpy()
        # compute rot mat that'll align the hand normal with target direction
        _align_rotmat = rotation_matrix_from_vectors(og_hand_normal, target_hand_normal)
        align_tf = np.eye(4)
        align_tf[:3, :3] = _align_rotmat
        align_tf[:3, 3] = target_pos.detach().clone().cpu().numpy()
        # NOTE: align_tf gives us 1 possible pose.
        # Now using the same hand normal direction, we rotate around Z axis to get other rotations
        num_rotations = self.num_particles
        angles = np.linspace(0, 2 * np.pi, num_rotations, endpoint=False)
        rotations = R.from_rotvec(angles[:, np.newaxis] * og_hand_normal)
        gripper_poses = np.zeros((num_rotations, 4, 4))
        aligned_tfs = np.zeros((num_rotations, 4, 4))

        for i, _r in enumerate(rotations):
            rot_tf = np.eye(4)
            rot_tf[:3, :3] = _r.as_matrix()
            aligned_tfs[i] = align_tf @ rot_tf
            align_7d = convert_4x4_to_7dpose(aligned_tfs[i])
            palm_pose_7d = convert_aligned_to_gripper_pose(align_7d, self.robot_name)
            palm_pose_tf = convert_7dpose_to_4x4(palm_pose_7d)
            gripper_poses[i] = palm_pose_tf

        random_rot = torch.tensor(gripper_poses[:, :3, :3], device=self.device).float()

        palm_posns = torch.tensor(gripper_poses[:, :3, 3], device=self.device).float()

        # NOTE: For a rotmat R, R.T.reshape(9)[:6] will give the ortho6d elements in order
        # I.e x1,x2,x3,y1,y2,y3 instead of (x1,y1,z1,x2,y2,z2)!!!
        self.q_current[:, 3:9] = random_rot.transpose(1, 2).reshape(
            self.num_particles, 9
        )[:, :6]
        self.q_current[:, :3] = palm_posns

        # DOFs initialization
        # Set the dof values to be initialized between (lower, lower + range * rand_0_1 * scale)
        self.q_current[:, 9:] = (
            self.init_random_scale
            * torch.rand_like(self.q_current[:, 9:])
            * (self.q_joint_upper - self.q_joint_lower)
            + self.q_joint_lower
        )
        self.q_current.requires_grad = True
        self.optimizer = torch.optim.Adam([self.q_current], lr=self.learning_rate)

        # # NOTE: DEBUG INFO for pose initialization
        # pose_init_data = {
        #     "tfs_grp": gripper_poses,
        #     "tfs_alg": aligned_tfs,
        #     "avg_norm_dir": avg_norm_dir.clone().detach().cpu().numpy(),
        #     "pts_mean": avg_palm_pts.detach().clone().cpu().numpy(),
        #     "pts_norm": avg_norm_dir.detach().clone().cpu().numpy(),
        #     "target_pt": target_pos.detach().clone().cpu().numpy(),
        #     "target_hand_normal": target_hand_normal,
        #     "q_current": self.q_current[:, :9].detach().clone().cpu().numpy(),
        # }
        # return pose_init_data

    def compute_energy_euclidean_dist(self):
        hand_surface_points_ = self.handmodel.get_surface_points()
        # Use all pts for penetration energy, use correspondence pts for distance energy
        hand_surface_points_corr = hand_surface_points_.clone()[:, self.grp_corr_idxs]
        obj_pts_corr = self.object_point_cloud[self.obj_corr_mask]
        num_particles = self.num_particles

        ## Compute Eucledian distance between the corresponding hand-object pairs
        ## NOTE: They are already in order!
        npts_object = obj_pts_corr.size()[0]
        npts_hand = hand_surface_points_corr.size()[1]
        assert npts_hand == npts_object  # due to correspondence

        batch_object_point_cloud = obj_pts_corr.unsqueeze(0).repeat(num_particles, 1, 1)

        hand_obj_dist = (hand_surface_points_corr - batch_object_point_cloud).norm(
            dim=2
        )  # shape (B, N)
        energy_contact = hand_obj_dist.mean(dim=1)  # shape (B, )

        ## Compute Penetration between hand object points
        npts_hand = hand_surface_points_.size()[1]
        npts_object = self.object_point_cloud.size()[0]
        batch_object_point_cloud = self.object_point_cloud.unsqueeze(0).repeat(
            self.num_particles, 1, 1
        )
        batch_object_point_cloud = batch_object_point_cloud.reshape(
            self.num_particles, 1, npts_object, 3
        )
        hand_surface_points = hand_surface_points_.reshape(
            self.num_particles, 1, npts_hand, 3
        )
        hand_surface_points = hand_surface_points.repeat(
            1, npts_object, 1, 1
        ).transpose(1, 2)
        batch_object_point_cloud = batch_object_point_cloud.repeat(1, npts_hand, 1, 1)
        hand_object_dist = (hand_surface_points - batch_object_point_cloud).norm(dim=3)
        hand_object_dist, hand_object_indices = hand_object_dist.min(dim=2)
        hand_object_points = torch.stack(
            [self.object_point_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_normal = torch.stack(
            [self.object_normal_cloud[x, :] for x in hand_object_indices], dim=0
        )
        hand_object_signs = (
            (hand_object_points - hand_surface_points_) * hand_object_normal
        ).sum(dim=2)
        hand_object_signs = (hand_object_signs > 0).float()
        energy_penetration = (hand_object_signs * hand_object_dist).mean(dim=1)

        energy = energy_contact + self.penetration_energy_weight * energy_penetration
        self.energy = energy

        # TODO: add a normalized energy
        z_norm = F.relu(self.q_current[:, 9:] - self.q_joint_upper) + F.relu(
            self.q_joint_lower - self.q_current[:, 9:]
        )
        self.energy = energy + z_norm.sum(dim=1)
        if self.verbose_energy:
            return energy, energy_penetration, z_norm
        else:
            return energy

    def step(self):
        self.optimizer.zero_grad()
        self.handmodel.update_kinematics(q=self.q_current)
        energy = self.compute_energy()
        energy.mean().backward()
        self.optimizer.step()
        self.global_step += 1

    def get_opt_q(self):
        return self.q_current.detach()

    def set_opt_q(self, opt_q):
        self.q_current.copy_(opt_q.detach().to(self.device))

    def get_plotly_data(self, index=0, color="pink", opacity=0.7):
        # self.handmodel.update_kinematics(q=self.q_current)
        return self.handmodel.get_plotly_data(
            q=self.q_current, i=index, color=color, opacity=opacity
        )


class AdamGrasp:
    def __init__(
        self,
        robot_name,
        writer,
        contact_map_goal=None,
        num_particles=32,
        init_rand_scale=0.5,
        max_iter=300,
        steps_per_iter=2,
        learning_rate=5e-3,
        device="cuda",
        energy_func_name="align_dist",
    ):
        self.writer = writer
        self.robot_name = robot_name
        self.contact_map_goal = contact_map_goal
        self.num_particles = num_particles
        self.init_rand_scale = init_rand_scale
        self.learning_rate = learning_rate
        self.device = device
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.energy_func_name = energy_func_name

        self.opt_model = CMapAdam(
            robot_name=robot_name,
            contact_map_goal=None,
            num_particles=self.num_particles,
            init_rand_scale=init_rand_scale,
            learning_rate=learning_rate,
            energy_func_name=self.energy_func_name,
            device=device,
        )

    def run_adam(self, object_name, contact_map_goal, running_name):
        q_trajectory = []
        self.opt_model.reset(contact_map_goal, running_name, self.energy_func_name)
        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())
        iters_per_print = self.max_iter // 2
        for i_iter in tqdm(range(self.max_iter), desc=f"{running_name}"):
            self.opt_model.step()
            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())
            if i_iter % iters_per_print == 0 or i_iter == self.max_iter - 1:
                print(f"min energy: {self.opt_model.energy.min(dim=0)[0]:.4f}")
                print(f"min energy index: {self.opt_model.energy.min(dim=0)[1]}")
            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {
                    f"{i_energy}": energy[i_energy] for i_energy in range(len(energy))
                }
                self.writer.add_scalars(
                    main_tag=f"energy/{running_name}",
                    tag_scalar_dict=tag_scaler_dict,
                    global_step=i_iter,
                )
                self.writer.add_scalar(
                    tag=f"index/{running_name}",
                    scalar_value=energy.index(min(energy)),
                    global_step=i_iter,
                )
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        return (
            q_trajectory,
            self.opt_model.energy.detach().cpu().clone(),
            self.steps_per_iter,
        )


class GcsAdamGrasp:
    def __init__(
        self,
        robot_name,
        writer,
        contact_map_goal=None,
        num_particles=32,
        max_iter=300,
        steps_per_iter=2,
        learning_rate=5e-3,
        device="cuda",
        energy_func_name="euclidean_dist",
        palm_coords_u_upper=0.3,
        palm_coords_v_lower=0.6,
        verbose=False,
        num_print_per_opt=1,
    ):
        gdx_gname = lambda x: "robotiq_3finger_gdx" if x == "robotiq_3finger" else x

        self.writer = writer
        self.robot_name = robot_name
        self.contact_map_goal = contact_map_goal
        self.num_particles = num_particles
        self.learning_rate = learning_rate
        self.device = device
        self.max_iter = max_iter
        self.steps_per_iter = steps_per_iter
        self.energy_func_name = energy_func_name
        self.verbose = verbose
        self.num_print_per_opt = num_print_per_opt

        self.opt_model = GcsHandOpt(
            robot_name=gdx_gname(robot_name),
            contact_map_goal=None,
            num_particles=self.num_particles,
            energy_func_name=energy_func_name,
            penetration_energy_weight=60,
        )

    def run_adam(
        self,
        object_name,
        contact_map_goal,
        running_name,
        palm_u_upper=0.3,
        palm_v_lower=0.6,
    ):
        q_trajectory = []

        self.opt_model.reset(
            contact_map_goal,
            running_name,
            self.energy_func_name,
            ub_theta=palm_u_upper,
            lb_phi=palm_v_lower,
        )

        with torch.no_grad():
            opt_q = self.opt_model.get_opt_q()
            q_trajectory.append(opt_q.clone().detach())
        iters_per_print = int(
            self.max_iter // max(1.5, self.num_print_per_opt)
        )  # i.e print atleast once per optimization

        # for i_iter in tqdm(range(self.max_iter), desc=f"{running_name}"):
        for i_iter in range(self.max_iter):
            self.opt_model.step()

            with torch.no_grad():
                opt_q = self.opt_model.get_opt_q()
                q_trajectory.append(opt_q.clone().detach())

            if self.verbose and (
                (i_iter + 1) % iters_per_print == 0 or i_iter == self.max_iter - 1
            ):
                print(
                    f"{running_name}: min energy: {self.opt_model.energy.min(dim=0)[0]:.4f}"
                )
                print(
                    f"{running_name}: min energy index: {self.opt_model.energy.min(dim=0)[1]}"
                )

            with torch.no_grad():
                energy = self.opt_model.energy.detach().cpu().tolist()
                tag_scaler_dict = {
                    f"{i_energy}": energy[i_energy] for i_energy in range(len(energy))
                }
                self.writer.add_scalars(
                    main_tag=f"energy/{running_name}",
                    tag_scalar_dict=tag_scaler_dict,
                    global_step=i_iter,
                )
                self.writer.add_scalar(
                    tag=f"index/{running_name}",
                    scalar_value=energy.index(min(energy)),
                    global_step=i_iter,
                )
        q_trajectory = torch.stack(q_trajectory, dim=0).transpose(0, 1)
        return (
            q_trajectory,
            self.opt_model.energy.detach().cpu().clone(),
            self.steps_per_iter,
        )
