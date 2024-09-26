from torch.utils.data import DataLoader
import lightning as L
from typing import Any

from .gdx_dataset import GdxGrpCoordsDataset


class GdxDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        robot_name_list,
        data_type: str = "cmap+gcs",
    ):
        super().__init__()
        assert data_type in {"cmap+gcs", "cmap"}
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.robot_name_list = robot_name_list
        print(f"NOTE: Training using {data_type} dataset!\n")
        if data_type != "cmap+gcs":
            raise NotImplementedError
        self.dset_class = GdxGrpCoordsDataset

    def setup(self, stage):
        self.train_dataset = self.dset_class(
            dataset_basedir=self.data_dir,
            mode="train",
            robot_name_list=self.robot_name_list,
        )
        self.val_dataset = self.dset_class(
            dataset_basedir=self.data_dir,
            mode="validate",
            robot_name_list=self.robot_name_list,
        )
        # No test data split in GenDexGrasp paper dataset!

    def train_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )

    def test_dataloader(self) -> Any:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
        )

