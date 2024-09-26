import os
import sys
import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np
from scipy.spatial.distance import cdist

sys.path.append("..")
from utils.grasp_utils import get_handmodel


def get_obj_coords(obj_pts, grp_pts, grp_coords, cmap):
    """
    Get GCS coordinates for object pts that are in close contact with the gripper hand.

    Input:
     obj_pts: shape (N, 3) numpy float array for object point cloud
     grp_pts: shape (M, 3) numpy float array for gripper surface points
     grp_coords: shape (M, 2) torch tensor gripper points coordinates in normalized space
     cmap: (Optional) shape (N, 1) numpy float array giving which object pts are close to the gripper (high cmap value, close to 1). Can be used to filter out object surface points quickly

     Returns:
      obj_coords: shape (N, 2) numpy float array. All object points get a coordinate value, those not in contact (low cmap value) get a default no-contact coordinate (0, 0)
    """

    obj_hand_dist = torch.cdist(obj_pts, grp_pts)
    closest_dists, min_idxs = torch.min(obj_hand_dist, dim=1)
    obj_coords = grp_coords[min_idxs].clone().detach()
    assert obj_coords.shape[0] == obj_pts.shape[0]

    # no_contact_idxs = closest_dists > 0.05
    no_contact_idxs = cmap[:, 0] < 0.008  # cmap threshold!!
    obj_coords[no_contact_idxs] = (cmap[no_contact_idxs] / 2).to(obj_coords.device)
    ### obj_coords[no_contact_idxs] = 0
    ### Return a cpu tensor for storing to disk
    return obj_coords.detach().cpu()


def make_parser():
    parser = argparse.ArgumentParser(
        prog="ComputeGcsGdx",
        description="Compute gripper coordinates mapping for each grasp in GenDexGrasp Dataset",
    )
    parser.add_argument(
        "-d",
        "--dset_root",
        type=str,
        default="/home/ninad/Datasets/GenDexGrasp/",
        help="Path to the GenDexGrasp Dataset directory",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/CMapDataset-sqrt_align",
        help="grasps directory",
    )
    parser.add_argument(
        "--objects_file",
        type=str,
        default="object_pts_and_normals.pt",
        help="filename for object point clouds an normals",
    )
    parser.add_argument(
        "-m",
        "--grasps_file",
        type=str,
        default="cmap_dataset.pt",
        help="filename for grasps and contactmaps",
    )
    parser.add_argument(
        "-t",
        "--cmap_thresh",
        type=float,
        default="0",
        help="threshold for no contact and hence will get default (0, 0) coordinates",
    )
    parser.add_argument(
        "-n",
        "--num_jobs",
        type=int,
        default=8,
        help="Number of processing jobs to run in parallel",
    )
    return parser


def main(args):
    dset_root = args.dset_root
    data_dir = args.data_dir
    object_data_fname = args.objects_file
    grasp_data_fname = args.grasps_file
    cmap_threshold = args.cmap_thresh
    num_jobs = args.num_jobs

    GRIPPERS = {"allegro", "barrett", "ezgripper", "robotiq_3finger", "shadowhand"}
    gdx_gname = lambda x: "robotiq_3finger_gdx" if x == "robotiq_3finger" else x

    gripper_models = {
        gname: get_handmodel(
            robot=gdx_gname(gname),
            batch_size=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            hand_scale=1.0,
            json_path="data/urdf/urdf_assets_meta.json",
            datadir=dset_root,
        )
        for gname in GRIPPERS
    }
    print("\nLoaded Gripper Models\n")

    cmap_dset = torch.load(os.path.join(dset_root, data_dir, grasp_data_fname))
    meta = cmap_dset["metadata"]
    obj_pts_normals = torch.load(os.path.join(dset_root, data_dir, object_data_fname))
    print("Loaded Cmap dataaset and Object Point clouds.\n")

    cmap_gcs_dset = {}
    cmap_gcs_dset["info"] = deepcopy(cmap_dset["info"])
    meta_gcs = []
    for gdata in tqdm(meta):
        curr_cmap, curr_grasp, curr_obj, curr_hand = gdata
        curr_handmodel = gripper_models[curr_hand]
        obj_pts = (
            obj_pts_normals[curr_obj][:, :3].clone().detach().to(curr_handmodel.device)
        )
        grp_pts = (
            curr_handmodel.get_surface_points(
                q=curr_grasp.unsqueeze(0).to(curr_handmodel.device)
            )[0]
            .clone()
            .detach()
        )
        grp_coords = curr_handmodel.get_gripper_coords().clone().detach()
        obj_coords = get_obj_coords(
            obj_pts,
            grp_pts,
            grp_coords,
            curr_cmap,
        )

        result_tuple = (
            curr_cmap.clone(),
            curr_grasp.clone().to(curr_cmap.device),  # put this also on cpu
            curr_obj,
            curr_hand,
            obj_coords,
        )
        meta_gcs.append(result_tuple)

    cmap_gcs_dset["metadata"] = meta_gcs
    out_file = os.path.join(dset_root, data_dir, "cmap_gcs_dataset.pt")
    torch.save(cmap_gcs_dset, out_file)
    # Done
    print("Done!")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    main(args)
