import torch
import torch.nn as nn
from torch.nn import functional as F


class GcsLoss(nn.Module):
    def __init__(
        self,
        lw_init_recon: float,
        lw_init_kld: float,
        ann_temp: float,
        ann_per_epochs: int = 1,
        attn_weight: float = 3,
    ):
        super(GcsLoss, self).__init__()
        self.lw_recon = lw_init_recon
        self.lw_kld = lw_init_kld
        self.ann_temp = ann_temp
        self.iter_counter = 0
        self.ann_per_epochs = ann_per_epochs
        self.use_attn = attn_weight > 1
        self.attn_alpha = attn_weight

    def forward(self, means, logvars, uv_coords_gt, uv_coords_pred, cmap_gt=None):
        """
        :param means:
        :param logvars:
        :param uv_coords_gt: (B, N, 2+)
        :param uv_coords_pred: (B, N, 2+)

        :return:
            loss, loss_reconstruction, loss_KL_Divergence
        """
        loss_kld = (
            -0.5 * torch.sum(1 + logvars - means.pow(2) - logvars.exp(), dim=-1).mean()
        )

        if self.use_attn and cmap_gt is not None:
            square_error = torch.square(uv_coords_gt - uv_coords_pred).sum(dim=-1)
            attention_weights = torch.exp(cmap_gt.squeeze(-1) * self.attn_alpha)
            square_error = square_error * attention_weights
            # now we (B, N) sum-> (B, ) div-> (B,) .mean() -> (scalar)
            loss_recon = torch.sqrt(
                (square_error.sum(dim=-1) / attention_weights.sum(dim=-1))
            ).mean()
        else:
            loss_recon = torch.sqrt(
                torch.square(uv_coords_gt - uv_coords_pred).sum(dim=-1)
            ).mean()

        loss = self.lw_kld * loss_kld + self.lw_recon * loss_recon
        return loss, loss_recon, loss_kld

    @staticmethod
    def metrics(self, uv_coords_gt, uv_coords_pred):
        return F.smooth_l1_loss(uv_coords_gt, uv_coords_pred, reduction="mean")

    def update_kld_weight(self):
        self.lw_kld *= self.ann_temp

    def apply_iter(self):
        # NOT USED with Pytorch Lightning mode
        if self.iter_counter % self.ann_per_epochs == self.ann_per_epochs - 1:
            self.lw_kld *= self.ann_temp
        self.iter_counter += 1
