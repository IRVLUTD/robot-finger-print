import os
import sys
from datetime import datetime
import time
import shutil
import math

from argparse import ArgumentParser

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from model.grasp_network import GcsGraspModel
from utils.dataset import GdxDataModule


def main(args, time_tag):
    # RNG Seeding
    seed_everything(args.seed, workers=True)

    # disable unseen robotic hand
    robot_name_list = [
        "ezgripper",
        "barrett",
        "robotiq_3finger",
        "allegro",
        "shadowhand",
    ]
    unseen_robots = []
    if args.disable_shadowhand:
        robot_name_list.remove("shadowhand")
        unseen_robots.append("shadowhand")
    if args.disable_allegro:
        robot_name_list.remove("allegro")
        unseen_robots.append("allegro")
    if args.disable_robotiq_3finger:
        robot_name_list.remove("robotiq_3finger")
        unseen_robots.append("robotiq_3finger")
    if args.disable_barrett:
        robot_name_list.remove("barrett")
        unseen_robots.append("barrett")
    if args.disable_ezgripper:
        robot_name_list.remove("ezgripper")
        unseen_robots.append("ezgripper")
    print(f"Robot name list: {robot_name_list}")
    print(f"Unseen robots:", unseen_robots)
    if len(unseen_robots) > 0:
        trn_dset_info = "dset_unseen_" + "_".join(unseen_robots)
    else:
        trn_dset_info = "dset_fullrobots"

    to_overfit = args.overfit
    if to_overfit:
        print("Overfitting RUN!!")
    num_overfit = (
        8 if to_overfit else 0
    )  # 0 means disable overfitting (default in Trainer)

    exp_time = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    exp_name = f"exp-{exp_time}-{trn_dset_info}"
    if to_overfit:
        exp_name += f"-overfit_{num_overfit}"

    exp_logger = TensorBoardLogger(
        save_dir="logs",
        name="gcs_gdx",
        version=exp_name,
    )

    data_dir = "dataset/GenDexGrasp/dataset/CMapDataset-sqrt_align"
    data_module = GdxDataModule(
        data_dir,
        args.batchsize if not to_overfit else math.floor(num_overfit / 4),
        robot_name_list,
        data_type="cmap+gcs",
    )
    grasp_model = GcsGraspModel(
        learning_rate=args.lr,
        cmap_loss_wrecon=args.lw_recon,
        cmap_loss_wkld=args.lw_kld,
        cmap_loss_temp=args.ann_temp,
        cmap_loss_ann_per_epoch=args.ann_per_epochs,
        pred_type=args.pred,
        loss_attn_weight=args.attn_alpha,
        decay_lr_freq=args.decay_lr,
    )

    val_every_n_epochs = 100 if to_overfit else 1
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=val_every_n_epochs,
        save_last=True,
        save_top_k=-1,  # <--- this is important! saves all ckpts
    )

    trainer = Trainer(
        logger=exp_logger,
        callbacks=[checkpoint_callback] if not to_overfit else None,
        max_epochs=500 if to_overfit else args.n_epochs,
        check_val_every_n_epoch=val_every_n_epochs,
        devices=args.devices,
        overfit_batches=num_overfit,
        log_every_n_steps=1 if to_overfit else 100 * math.floor(args.batchsize / 128),
    )

    if trainer.global_rank == 0:
        # Perform rank 0 specific operations, such as creating a log folder
        log_dir = os.path.join(
            exp_logger.save_dir, str(exp_logger.name), str(exp_logger.version)
        )
        os.makedirs(log_dir, exist_ok=True)
        # Save command used to invoke training
        with open(os.path.join(log_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv) + "\n")
            f.write("Seen robots: " + " ".join(robot_name_list) + "\n")
            f.write("Unseen robots: " + " ".join(unseen_robots) + "\n")

        print("LogDir:", log_dir)
        print("Creating folder for model checkpoints (weights)...")
        os.makedirs(os.path.join(log_dir, "checkpoints"))
        print("Copying src files to logdir...")
        os.makedirs(os.path.join(log_dir, "src"), exist_ok=True)
        for fn in os.listdir("."):
            if fn[-3:] == ".py":
                fn = os.path.join(fn)
                shutil.copy(fn, os.path.join(log_dir, "src", fn))
        src_dir_list = ["utils", "model"]
        for src_dir in src_dir_list:
            for fn in os.listdir(src_dir):
                if fn[-3:] == ".py":
                    fn = os.path.join(src_dir, fn)
                    os.makedirs(os.path.join(log_dir, "src", src_dir), exist_ok=True)
                    shutil.copy(fn, os.path.join(log_dir, "src", fn))

    print("Begin Training")
    trainer.fit(
        model=grasp_model,
        datamodule=data_module,
    )


def make_parser():
    parser = ArgumentParser()
    parser.add_argument("--comment", default="", type=str)
    parser.add_argument("--id", default=0, type=int)
    parser.add_argument(
        "--pred", default="gcs", help="Prediction type: ['gcs', 'cmap' or 'gcs+cmap']"
    )
    parser.add_argument("--devices", default=1, type=int, help="number of gpu devices")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="overfit on a small number of batches to check",
    )

    parser.add_argument("--batchsize", default=64, type=int)
    parser.add_argument("--n_epochs", default=72, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lw_recon", default=1000.0, type=float)  # sqrt(MSE(x, y))
    parser.add_argument("--lw_kld", default=0.01, type=float)
    parser.add_argument("--ann_temp", default=1.0, type=float)
    parser.add_argument("--ann_per_epochs", default=4, type=int)
    parser.add_argument("--attn_alpha", type=float, default=3, help="loss attn alpha")
    parser.add_argument(
        "--decay_lr",
        type=float,
        default=1000,
        help="epoch frequency to decay LR. default is 1000 epochs to disable decaying!",
    )
    parser.add_argument("--seed", type=int, default=42, help="randomization seed")

    parser.add_argument("--disable_shadowhand", default=False, action="store_true")
    parser.add_argument("--disable_allegro", default=False, action="store_true")
    parser.add_argument("--disable_robotiq_3finger", default=False, action="store_true")
    parser.add_argument("--disable_barrett", default=False, action="store_true")
    parser.add_argument("--disable_ezgripper", default=False, action="store_true")

    # parser.add_argument('--enable_only_barrett', default=False, action='store_true')
    return parser


if __name__ == "__main__":
    parser = make_parser()
    start_time = time.time()
    time_tag = start_time
    parser = make_parser()
    args = parser.parse_args()
    print("[INFO] Arguments passed:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args, time_tag)
    print("finish training...")
    print(f"consuming time: {time.time() - start_time}")
