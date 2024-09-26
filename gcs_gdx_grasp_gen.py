import argparse
import json
import os.path
import time
import sys
import shutil
from datetime import datetime
from tqdm import tqdm
import pickle
from collections import defaultdict

import trimesh as tm
import plotly.graph_objects as go

import torch
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch import seed_everything as set_global_seed


from model.hand_opt import GcsAdamGrasp
from utils.viz import plot_point_cloud_cmap, plot_mesh_from_name


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_name", default="ezgripper", type=str)

    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="log dir relative to repo root. Example: `logs/gcs_gdx/exp-timestamp-dset/`...",
    )
    parser.add_argument(
        "--inf_dir",
        type=str,
        required=True,
        help="cmap inf dir relative to checkpoint dir. Will likely be of the format: 'inference-[ckpt]-[pre_process]-[time_tag]-[comment]'",
    )
    parser.add_argument(
        "--dataset",
        default="fullrobots",
        type=str,
        help="Can be from: ['fullrobots', 'unseen_[robot_name]]",
    )
    parser.add_argument(
        "--dataset_id",
        default="sharp_lift",
        type=str,
        help="func used for post-processing predicted contact maps",
    )
    parser.add_argument(
        "--domain",
        default="ood",
        type=str,
        help="whether ood objects or in-domain (id)",
    )
    parser.add_argument("--energy_func", default="euclidean_dist", type=str)
    parser.add_argument("--comment", default="gen", type=str)

    parser.add_argument("--max_iter", default=100, type=int)
    parser.add_argument("--steps_per_iter", default=1, type=int)
    parser.add_argument("--num_particles", default=32, type=int)
    parser.add_argument("--learning_rate", default=5e-3, type=float)
    parser.add_argument("--init_rand_scale", default=0.5, type=float)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output during grasp optimization",
    )

    args_ = parser.parse_args()
    tag = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    return args_, tag


if __name__ == "__main__":
    set_global_seed(seed=42)
    torch.set_printoptions(precision=4, sci_mode=False, edgeitems=8)
    args, time_tag = get_parser()
    print(args)

    # TODO: Add this later
    assert args.dataset in {
        "fullrobots",
        "unseen_shadowhand",
        "unseen_barrett",
        "unseen_ezgripper",
    }
    assert args.dataset in args.logdir  # check if the checkpoint being used is correct?

    datadir = "./dataset/GenDexGrasp"
    object_mesh_basedir = os.path.join(datadir, "data/object")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    robot_name = args.robot_name

    cmapinf_dir = os.path.join(
        args.logdir,
        args.inf_dir,
    )
    try:
        cmap_dataset = torch.load(os.path.join(cmapinf_dir, f"cmap_{args.domain}.pt"))
    except:
        raise FileNotFoundError("Error occured when loading inference data...")

    with open(
        os.path.join(cmapinf_dir, "split_train_validate_objects.json"), "rb"
    ) as f:
        object_split_info = json.load(f)
    if args.domain == "ood":
        object_name_list = object_split_info["validate"]
    elif args.domain == "id":
        object_name_list = object_split_info["train"]
    else:
        raise NotImplementedError

    ##### Setup Log Dir ####
    graspgen_logs_basedir = os.path.join(
        "logs",
        "graspgen_gcs_gdx",
        f"{args.dataset}-{args.dataset_id}",
        f"{args.domain}-{args.robot_name}-{args.energy_func}-{args.comment}",
    )
    tb_dir = os.path.join(graspgen_logs_basedir, "tb_dir")
    tra_dir = os.path.join(graspgen_logs_basedir, "tra_dir")
    viz_dir = os.path.join(graspgen_logs_basedir, "viz_dir")
    os.makedirs(graspgen_logs_basedir, exist_ok=True)
    os.makedirs(tra_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    with open(os.path.join(graspgen_logs_basedir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    print("Copying src files to logdir...")
    src_dir_list = ["utils", "model"]
    os.makedirs(os.path.join(graspgen_logs_basedir, "src"), exist_ok=True)
    for fn in os.listdir("."):
        if fn[-3:] == ".py":
            shutil.copy(fn, os.path.join(graspgen_logs_basedir, "src", fn))
    for src_dir in src_dir_list:
        for fn in os.listdir(f"{src_dir}"):
            os.makedirs(
                os.path.join(graspgen_logs_basedir, "src", f"{src_dir}"), exist_ok=True
            )
            if fn[-3:] == ".py" or fn[-5:] == ".yaml":
                shutil.copy(
                    os.path.join(f"{src_dir}", fn),
                    os.path.join(graspgen_logs_basedir, "src", f"{src_dir}", fn),
                )
    ##### Log Dir Setup Done ####

    #### Init Hand Model
    goptmodel = GcsAdamGrasp(
        robot_name=robot_name,
        writer=writer,
        contact_map_goal=None,
        num_particles=args.num_particles,
        max_iter=args.max_iter,
        steps_per_iter=args.steps_per_iter,
        learning_rate=args.learning_rate,
        device=device,
        energy_func_name=args.energy_func,
        verbose=True,
        # verbose=args.verbose,
    )

    for obj_name in sorted(object_name_list):
        print(f"Grasp opt for {robot_name} and {obj_name}...")
        obj_cmap_dataset = cmap_dataset[obj_name]
        grp_obj_gopt_data = {}
        grp_obj_gopt_data["gripper_name"] = robot_name
        grp_obj_gopt_data["obj_name"] = obj_name
        grp_obj_gopt_data["num_samples"] = len(obj_cmap_dataset)
        grp_obj_gopt_data["dataset"] = args.dataset
        grp_obj_gopt_data["steps_per_iter"] = args.steps_per_iter
        grp_obj_gopt_data["q_data"] = []  # init with empty list
        grp_obj_gopt_data["energy"] = []  # init with empty list
        grp_obj_gopt_data["best_idx"] = []
        grp_obj_gopt_data["all_q_final"] = (
            []
        )  # init with empty list, also save full trajectory q vals

        for i_data in tqdm(obj_cmap_dataset):  # num_cmap_preds_per_object
            object_name = i_data["object_name"]
            object_point_cloud = i_data["object_point_cloud"]
            i_sample = i_data["i_sample"]
            contact_map_value = i_data["contact_map_value"]
            with torch.no_grad():
                contact_map_value = torch.clamp(contact_map_value, min=0, max=1)
            running_name = f"{robot_name}-{object_name}-{i_sample}"
            contact_map_goal = torch.cat(
                [object_point_cloud, contact_map_value], dim=1
            ).to(device)

            record = goptmodel.run_adam(
                object_name=object_name,
                contact_map_goal=contact_map_goal,
                running_name=running_name,
                palm_u_upper=0.3,
                palm_v_lower=0.5,
            )

            with torch.no_grad():
                q_tra, energy, steps_per_iter = record
                best_energy, best_idx = energy.min(dim=0)
                best_q = q_tra[best_idx.item(), -1:]  # shape (1, 9 + n_dofs)

                grp_obj_gopt_data["q_data"].append(best_q)
                grp_obj_gopt_data["energy"].append(best_energy.item())  # scalar value
                grp_obj_gopt_data["best_idx"].append(best_idx.item())
                grp_obj_gopt_data["all_q_final"].append(
                    q_tra[:, -1].unsqueeze(0).detach().clone()
                )  # shape(32, 9 + n_dofs)

                # VIZ Data
                vis_data = goptmodel.opt_model.handmodel.get_plotly_data(
                    q=best_q.cuda(), color="pink"
                )
                vis_data += [
                    plot_point_cloud_cmap(
                        contact_map_goal[:, :3].detach().cpu().numpy(),
                        contact_map_goal[:, -1].detach().cpu().numpy(),
                    )
                ]
                vis_data += [plot_mesh_from_name(object_name, opacity=0.7)]
                fig = go.Figure(data=vis_data)
                fig.write_html(
                    os.path.join(
                        viz_dir, f"viz_gopt-{robot_name}-{object_name}-{i_sample}.html"
                    )
                )
        # convert the q and energy lists to tensor of shape (num_samples, -1)
        grp_obj_gopt_data["q_data"] = torch.cat(grp_obj_gopt_data["q_data"], dim=0)
        grp_obj_gopt_data["energy"] = torch.tensor(grp_obj_gopt_data["energy"])
        grp_obj_gopt_data["all_q_final"] = torch.cat(
            grp_obj_gopt_data["all_q_final"], dim=0
        )

        torch.save(
            grp_obj_gopt_data,
            os.path.join(tra_dir, f"gopt_result-{robot_name}-{object_name}.pt"),
        )
        print("\n")
