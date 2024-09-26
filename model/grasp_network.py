import lightning as L
import torch

from .modules import GcsCVAE
from .loss import GcsLoss


class GcsGraspModel(L.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        cmap_loss_wrecon: float = 100.0,
        cmap_loss_wkld: float = 0.1,
        cmap_loss_temp: float = 1.5,
        cmap_loss_ann_per_epoch: int = 2,
        pred_type: str = "gcs",
        loss_attn_weight: float = 3,
        decay_lr_freq: int = 1000,
    ):
        super().__init__()
        assert pred_type in {"cmap", "gcs", "gcs+cmap", "gcs_u", "gcs_v"}

        self.lr = learning_rate
        self.loss_criterion = GcsLoss(
            cmap_loss_wrecon,
            cmap_loss_wkld,
            cmap_loss_temp,
            cmap_loss_ann_per_epoch,
            loss_attn_weight,
        )

        self.pred_type = pred_type
        self.decay_lr_freq = decay_lr_freq
        if pred_type == "gcs":
            input_size = 5  # B,N,3 pts  and N, 2 gcs uvmap
            output_size = (
                64  # B, N, 64 global feats --> gets mapped to B, N, 2 uv preds
            )
        else:
            raise NotImplementedError

        self.model = GcsCVAE(
            encoder_layers_size=[input_size, 64, 128, 512, 512],
            decoder_decoder_layers_size=[64 + 512 + 128, 512, 64],
            uv_layers_size=[64, 32, 1],
        )
        self.cmap_loss_ann_per_epoch = cmap_loss_ann_per_epoch

        print(
            "Hparams:",
            decay_lr_freq,
            loss_attn_weight,
            cmap_loss_ann_per_epoch,
            cmap_loss_temp,
            learning_rate,
        )
        self.save_hyperparameters()

    def get_inputs(self, batch):
        input_pc = batch["full_pc"]  # (B, N, 3)
        target_cmap = batch["cmaps_fullpc"]  # (B, N, 1)
        target_gcs = batch["gcs_fullpc"]  # (B, N, 2)

        if self.pred_type == "cmap":
            target_map = target_cmap
        elif self.pred_type == "gcs_u":
            target_map = target_gcs[:, :, :1]
        elif self.pred_type == "gcs_v":
            target_map = target_gcs[:, :, 1:]
        elif self.pred_type == "gcs":
            target_map = target_gcs
        elif self.pred_type == "gcs+cmap":
            target_map = torch.cat(
                (
                    target_gcs,
                    target_cmap.unsqueeze(-1) if target_cmap.ndim == 2 else target_cmap,
                ),
                dim=-1,
            )
        return (input_pc, target_map, target_cmap, target_gcs)

    def forward(self, input_pc, gt_gcs):
        return self.model(input_pc, gt_gcs)

    def training_step(self, batch, batch_idx):

        input_pc, target_map, target_cmap, target_gcs = self.get_inputs(batch)

        _, pred_map, means, logvars, z_latent_code = self.model(input_pc, target_map)

        # For training use the attention weights dictated by the gt cmap
        loss_cmap, loss_recon, loss_kld = self.loss_criterion(
            means, logvars, target_map, pred_map, target_cmap
        )

        loss = loss_cmap
        loss_dict = {
            "loss/trn/overall": loss.item(),
            "loss/trn/cmap_rec": loss_recon.item(),
            "loss/trn/cmap_kld": loss_kld.item(),
        }
        self.log_dict(
            loss_dict,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log("trn_loss", loss_recon.item(), prog_bar=True)
        if batch_idx == 0:
            self._log_prediction_sample(
                input_pc,
                target_map,
                pred_map,
                gt_attn=None,
                mode="trn",
            )
        return loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.cmap_loss_ann_per_epoch == 0:
            self.loss_criterion.update_kld_weight()

    def validation_step(self, batch, batch_idx):
        input_pc, target_map, target_cmap, target_gcs = self.get_inputs(batch)
        _, pred_map, means, logvars, z_latent_code = self.model(input_pc, target_map)

        loss_cmap, loss_recon, loss_kld = self.loss_criterion(
            means, logvars, target_map, pred_map, target_cmap
        )

        loss = loss_cmap

        loss_dict = {
            "loss/val/overall": loss.item(),
            "loss/val/cmap_rec": loss_recon.item(),
            "loss/val/cmap_kld": loss_kld.item(),
        }
        self.log_dict(
            loss_dict,
            logger=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def test_step(self, batch, batch_idx):
        input_pc, target_map, target_cmap, target_gcs = self.get_inputs(batch)
        _, pred_map, means, logvars, z_latent_code = self.model(input_pc, target_map)
        loss_cmap, loss_recon, loss_kld = self.loss_criterion(
            means, logvars, target_map, pred_map
        )
        loss = loss_cmap

        loss_dict = {
            "loss/tst/overall": loss,
            "loss/tst/cmap_rec": loss_recon,
            "loss/tst/cmap_kld": loss_kld,
        }
        self.log_dict(loss_dict, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        input_pc, _ = self.get_inputs(batch)
        return self.model.predict(input_pc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_lr_freq, gamma=0.2
        )
        return [optimizer], [lr_scheduler]
