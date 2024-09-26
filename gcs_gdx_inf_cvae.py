import argparse
import json
import os
import sys
from datetime import datetime
from tqdm import tqdm

import trimesh as tm
import trimesh.sample

import torch
import torch.nn as nn
from lightning.pytorch import seed_everything

from model.grasp_network import GcsGraspModel
from utils.dataset import GdxDataModule

import plotly.graph_objects as go
from utils.viz import plot_point_cloud, plot_point_cloud_cmap, plot_mesh_from_name


def get_parser():
    parser = argparse.ArgumentParser()

    # Example Args for debug:  --logdir logs/gcs_gdx/exp-24-09-08_04-00-35 --ckpt epoch=5-step=984.ckpt

    parser.add_argument("--logdir", type=str, required=True)
    parser.add_argument(
        "--ckpt",
        type=str,
        help="checkpoint file name relative to 'logdir/checkpoints' dir",
        required=True,
    )

    parser.add_argument("--pre_process", default="sharp_lift", type=str)
    parser.add_argument("--num_per_seen_object", default=4, type=int)
    parser.add_argument("--num_per_unseen_object", default=16, type=int)
    parser.add_argument("--comment", default="debug", type=str)

    args_ = parser.parse_args()
    tag = datetime.now().strftime("%y%m%d_%H%M%S")
    return args_, tag


def pre_process_sharp_clamp(contact_map):
    # gap_th = 0.5  # delta_th = (1 - gap_th)
    # gap_th = min(contact_map.max().item(), gap_th)
    # delta_th = 1 - gap_th
    # contact_map[contact_map > 0.4] += delta_th
    # # contact_map += delta_th
    # contact_map = torch.clamp_max(contact_map, 1.0)
    return contact_map


def identity_map(contact_map):
    return contact_map


if __name__ == "__main__":
    seed_everything(42)
    args, time_tag = get_parser()

    pre_process_map = {"sharp_lift": pre_process_sharp_clamp, "identity": identity_map}
    pre_process_contact_map_goal = pre_process_map[args.pre_process]

    basedir = "dataset/GenDexGrasp"  # symlinked to actual location on disk
    data_dir = os.path.join(basedir, "dataset/CMapDataset-sqrt_align")

    ckpt_file = os.path.join(args.logdir, "checkpoints", args.ckpt)
    if not os.path.exists(ckpt_file):
        print("Ckpt file path is invalid:", ckpt_file)
        print("[ERROR]! Will exit...")
        exit(0)

    logs_basedir = os.path.join(
        args.logdir,
        f"inference-{args.ckpt}-{args.pre_process}-{args.comment}-{time_tag}",
    )

    vis_id_dir = os.path.join(logs_basedir, "vis_id_dir")
    vis_ood_dir = os.path.join(logs_basedir, "vis_ood_dir")
    cmap_path_id = os.path.join(logs_basedir, "cmap_id.pt")
    cmap_path_ood = os.path.join(logs_basedir, "cmap_ood.pt")
    os.makedirs(logs_basedir, exist_ok=False)
    os.makedirs(vis_id_dir, exist_ok=False)
    os.makedirs(vis_ood_dir, exist_ok=False)

    with open(os.path.join(logs_basedir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    device = "cuda"
    litmodel = GcsGraspModel.load_from_checkpoint(ckpt_file)
    litmodel.eval()
    model = litmodel.model

    with open(os.path.join(data_dir, "split_train_validate_objects.json"), "rb") as f:
        object_split_info = json.load(f)

    # Also save the split to inf dir
    with open(
        os.path.join(logs_basedir, "split_train_validate_objects.json"), "w"
    ) as f:
        json.dump(object_split_info, f)

    seen_object_list = object_split_info["train"]
    unseen_object_list = object_split_info["validate"]

    cmap_ood_all = {}
    for object_name in unseen_object_list:
        print(f"unseen object name: {object_name}")
        object_mesh: tm.Trimesh
        object_mesh = tm.load(
            os.path.join(
                basedir,
                "data/object",
                object_name.split("+")[0],
                object_name.split("+")[1],
                f'{object_name.split("+")[1]}.stl',
            )
        )
        cmap_ood_obj_list = []
        for i_sample in tqdm(range(args.num_per_unseen_object)):
            cmap_ood_sample = {
                "object_name": object_name,
                "i_sample": i_sample,
                "object_point_cloud": None,
                "contact_map_value": None,
            }
            # print(f"[{i_sample}/{args.num_per_unseen_object}] | {object_name}")

            object_point_cloud, faces_indices = trimesh.sample.sample_surface(
                mesh=object_mesh, count=2048
            )

            contact_points_normal = torch.tensor(
                object_mesh.face_normals[faces_indices]
            ).float()

            # contact_points_normal = torch.tensor(
            #     [object_mesh.face_normals[x] for x in faces_indices]
            # ).float()

            object_point_cloud = torch.tensor(object_point_cloud).float()
            object_point_cloud = torch.cat(
                [object_point_cloud, contact_points_normal], dim=1
            ).to(device)

            z_latent_code = torch.randn(1, model.latent_size, device=device).float()
            contact_map_value = model.inference(
                object_point_cloud[:, :3].unsqueeze(0), z_latent_code
            ).squeeze(0)

            ### process the contact map value
            # contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
            # contact_map_value = pre_process_contact_map_goal(contact_map_value).to(
            #     device
            # )

            contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1)

            cmap_ood_sample["object_point_cloud"] = object_point_cloud
            cmap_ood_sample["contact_map_value"] = contact_map_value
            cmap_ood_sample["latent_code"] = z_latent_code.squeeze(0)
            cmap_ood_obj_list.append(cmap_ood_sample)

            vis_data = []
            vis_data += [
                plot_point_cloud_cmap(
                    contact_map_goal[:, :3].cpu().detach().numpy(),
                    contact_map_goal[:, 6].cpu().detach().numpy(),
                )
            ]
            vis_data += [plot_mesh_from_name(f"{object_name}")]
            fig = go.Figure(data=vis_data)
            fig.write_html(
                os.path.join(vis_ood_dir, f"unseen-{object_name}-{i_sample}.html")
            )
        print("\n")
        cmap_ood_all[object_name] = cmap_ood_obj_list
    torch.save(cmap_ood_all, cmap_path_ood)

    cmap_id_all = {}
    for object_name in seen_object_list:
        print(f"seen object name: {object_name}")
        object_mesh: tm.Trimesh
        object_mesh = tm.load(
            os.path.join(
                basedir,
                "data/object",
                object_name.split("+")[0],
                object_name.split("+")[1],
                f'{object_name.split("+")[1]}.stl',
            )
        )
        cmap_id_obj = []
        for i_sample in tqdm(range(args.num_per_seen_object)):
            cmap_id_sample = {
                "object_name": object_name,
                "i_sample": i_sample,
                "object_point_cloud": None,
                "contact_map_value": None,
            }
            # print(f"[{i_sample}/{args.num_per_seen_object}] | {object_name}")

            object_point_cloud, faces_indices = trimesh.sample.sample_surface(
                mesh=object_mesh, count=2048
            )

            contact_points_normal = torch.tensor(
                object_mesh.face_normals[faces_indices]
            ).float()

            # contact_points_normal = torch.tensor(
            #     [object_mesh.face_normals[x] for x in faces_indices]
            # ).float()

            object_point_cloud = torch.tensor(object_point_cloud).float()
            object_point_cloud = torch.cat(
                [object_point_cloud, contact_points_normal], dim=1
            ).to(device)

            z_latent_code = torch.randn(1, model.latent_size, device=device).float()
            contact_map_value = model.inference(
                object_point_cloud[:, :3].unsqueeze(0), z_latent_code
            ).squeeze(0)

            ### process the contact map value
            # contact_map_value = contact_map_value.detach().cpu().unsqueeze(1)
            # contact_map_value = pre_process_contact_map_goal(contact_map_value).to(
            #     device
            # )
            contact_map_goal = torch.cat([object_point_cloud, contact_map_value], dim=1)

            cmap_id_sample["object_point_cloud"] = object_point_cloud
            cmap_id_sample["contact_map_value"] = contact_map_value
            cmap_id_sample["latent_code"] = z_latent_code.squeeze(0)
            cmap_id_obj.append(cmap_id_sample)

            vis_data = []
            vis_data += [
                plot_point_cloud_cmap(
                    contact_map_goal[:, :3].cpu().detach().numpy(),
                    contact_map_goal[:, 6].cpu().detach().numpy(),
                )
            ]
            vis_data += [plot_mesh_from_name(f"{object_name}")]
            fig = go.Figure(data=vis_data)
            fig.write_html(
                os.path.join(vis_id_dir, f"seen-{object_name}-{i_sample}.html")
            )
        print("\n")
        cmap_id_all[object_name] = cmap_id_obj
    torch.save(cmap_id_all, cmap_path_id)

    print("Done with Inference...")
