import os
import torch
import json
from torch.utils.data import Dataset


class GenDexGraspCMapDataset(Dataset):
    def __init__(
        self,
        dataset_basedir,
        object_npts=2048,
        enable_disturb=True,
        disturbance_sigma=0.001,  # meter
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode="train",
        robot_name_list=[
            "ezgripper",
            "barrett",
            "robotiq_3finger",
            "allegro",
            "shadowhand",
        ],
    ):
        self.device = device
        self.dataset_basedir = dataset_basedir
        self.object_npts = object_npts
        self.enable_disturb = enable_disturb
        self.disturbance_sigma = disturbance_sigma
        self.robot_name_list = robot_name_list

        print("loading cmap metadata....")
        cmap_dataset = torch.load(os.path.join(dataset_basedir, "cmap_dataset.pt"))
        self.metadata_info = cmap_dataset["info"]
        self.metadata = cmap_dataset["metadata"]
        print("loading object point clouds....")
        self.object_point_clouds = torch.load(
            os.path.join(dataset_basedir, "object_point_clouds.pt")
        )

        if mode == "train":
            self.object_list = json.load(
                open(
                    os.path.join(dataset_basedir, "split_train_validate_objects.json"),
                    "rb",
                )
            )[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == "validate":
            self.object_list = json.load(
                open(
                    os.path.join(dataset_basedir, "split_train_validate_objects.json"),
                    "rb",
                )
            )[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == "full":
            self.object_list = (
                json.load(
                    open(
                        os.path.join(
                            dataset_basedir, "split_train_validate_objects.json"
                        ),
                        "rb",
                    )
                )["train"]
                + json.load(
                    open(
                        os.path.join(
                            dataset_basedir, "split_train_validate_objects.json"
                        ),
                        "rb",
                    )
                )["validate"]
            )
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        else:
            raise NotImplementedError()
        print(f"object selection: {self.object_list}")

        self.datasize = len(self.metadata)
        print("finish loading dataset....")

    def __len__(self):
        return self.datasize

    def __getitem__(self, item):
        disturbance = torch.randn(self.object_npts, 3) * self.disturbance_sigma
        map_value = self.metadata[item][0]
        robot_name = self.metadata[item][3]
        object_name = self.metadata[item][2]
        contact_map = (
            self.object_point_clouds[object_name] + disturbance * self.enable_disturb
        )
        # contact_map = torch.cat([contact_map, map_value], dim=1).to(self.device)
        # contact_map = torch.cat([contact_map, map_value], dim=1)
        # return contact_map, robot_name, object_name

        # Just mock certain aspects of partial point cloud and its contact map
        meta = {
            "partial_pc": contact_map,
            "full_pc": contact_map,
            "cmaps_fullpc": map_value,
            "cmaps_ppc": map_value,
        }
        return meta


class GdxGrpCoordsDataset(Dataset):
    def __init__(
        self,
        dataset_basedir,
        object_npts=2048,
        enable_disturb=True,
        disturbance_sigma=0.001,  # meter
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode="train",
        robot_name_list=[
            "ezgripper",
            "barrett",
            "robotiq_3finger",
            "allegro",
            "shadowhand",
        ],
        cmap_threshold=0.0001,
    ):
        self.device = device
        self.dataset_basedir = dataset_basedir
        self.object_npts = object_npts
        self.enable_disturb = enable_disturb
        self.disturbance_sigma = disturbance_sigma
        self.robot_name_list = robot_name_list
        self.cmap_thresh = cmap_threshold  # threshold below which to give object pts a default (0,0) coordinate value

        print("loading cmap+gcs metadata....")
        cmap_gcs_dataset = torch.load(
            os.path.join(dataset_basedir, "cmap_gcs_dataset.pt")
        )
        self.metadata_info = cmap_gcs_dataset["info"]
        self.metadata = cmap_gcs_dataset["metadata"]
        print("loading object point clouds....")
        self.object_point_clouds = torch.load(
            os.path.join(
                dataset_basedir, "object_point_clouds.pt"
            )  # object_point_clouds.pt or object_pts_and_normals.pt
        )

        if mode == "train":
            self.object_list = json.load(
                open(
                    os.path.join(dataset_basedir, "split_train_validate_objects.json"),
                    "rb",
                )
            )[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == "validate":
            self.object_list = json.load(
                open(
                    os.path.join(dataset_basedir, "split_train_validate_objects.json"),
                    "rb",
                )
            )[mode]
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        elif mode == "full":
            self.object_list = (
                json.load(
                    open(
                        os.path.join(
                            dataset_basedir, "split_train_validate_objects.json"
                        ),
                        "rb",
                    )
                )["train"]
                + json.load(
                    open(
                        os.path.join(
                            dataset_basedir, "split_train_validate_objects.json"
                        ),
                        "rb",
                    )
                )["validate"]
            )
            self.metadata = [t for t in self.metadata if t[2] in self.object_list]
            self.metadata = [t for t in self.metadata if t[3] in self.robot_name_list]
        else:
            raise NotImplementedError()
        print(f"object selection: {self.object_list}")

        self.datasize = len(self.metadata)
        print("finish loading dataset....")

    def __len__(self):
        return self.datasize

    def __getitem__(self, item):
        disturbance = torch.randn(self.object_npts, 3) * self.disturbance_sigma
        object_name = self.metadata[item][2]
        objpc = (
            self.object_point_clouds[object_name][:, :3]
            + disturbance * self.enable_disturb
        )
        cmap_value = self.metadata[item][0]
        gcs_value = self.metadata[item][-1]

        meta = {
            "full_pc": objpc,  # shape (N, 3)
            "cmaps_fullpc": cmap_value,  # shape (N, 1)
            "gcs_fullpc": gcs_value,  # shape (N, 2)
        }
        return meta
