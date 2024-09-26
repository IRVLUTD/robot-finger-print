import torch
import torch.nn as nn

class PointNetCmapEncoder(nn.Module):
    def __init__(self, layers_size=[4, 64, 128, 512]):
        super(PointNetCmapEncoder, self).__init__()
        self.layers_size = layers_size
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()

        for i in range(len(layers_size) - 1):
            self.conv_layers.append(nn.Conv1d(layers_size[i], layers_size[i + 1], 1))
            self.bn_layers.append(nn.BatchNorm1d(layers_size[i + 1]))
            nn.init.xavier_normal_(self.conv_layers[-1].weight)

    def forward(self, x):
        # input: B * N * 4
        # output: B * latent_size
        x = x.transpose(1, 2)
        for i in range(len(self.conv_layers) - 1):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = self.activate_func(x)
        x = self.bn_layers[-1](self.conv_layers[-1](x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.layers_size[-1])
        return x


class PointNetCmapDecoder(nn.Module):
    def __init__(
        self,
        global_feat_size=512,
        latent_size=128,
        pointwise_layers_size=[3, 64, 64],
        global_layers_size=[64, 128, 512],
        decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1],
    ):
        super(PointNetCmapDecoder, self).__init__()
        assert global_feat_size == global_layers_size[-1]
        assert (
            decoder_layers_size[0]
            == latent_size + global_feat_size + pointwise_layers_size[-1]
        )

        self.global_feat_size = global_feat_size
        self.latent_size = latent_size
        self.pointwise_layers_size = pointwise_layers_size
        self.global_layers_size = global_layers_size
        self.decoder_layers_size = decoder_layers_size

        self.pointwise_conv_layers = nn.ModuleList()
        self.pointwise_bn_layers = nn.ModuleList()
        self.global_conv_layers = nn.ModuleList()
        self.global_bn_layers = nn.ModuleList()
        self.activate_func = nn.ReLU()

        for i in range(len(pointwise_layers_size) - 1):
            self.pointwise_conv_layers.append(
                nn.Conv1d(pointwise_layers_size[i], pointwise_layers_size[i + 1], 1)
            )
            self.pointwise_bn_layers.append(
                nn.BatchNorm1d(pointwise_layers_size[i + 1])
            )
            nn.init.xavier_normal_(self.pointwise_conv_layers[-1].weight)

        for i in range(len(global_layers_size) - 1):
            self.global_conv_layers.append(
                nn.Conv1d(global_layers_size[i], global_layers_size[i + 1], 1)
            )
            self.global_bn_layers.append(nn.BatchNorm1d(global_layers_size[i + 1]))
            nn.init.xavier_normal_(self.global_conv_layers[-1].weight)

        self.decoder_conv_layers = nn.ModuleList()
        self.decoder_bn_layers = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()

        for i in range(len(decoder_layers_size) - 1):
            self.decoder_conv_layers.append(
                nn.Conv1d(decoder_layers_size[i], decoder_layers_size[i + 1], 1)
            )
            self.decoder_bn_layers.append(nn.BatchNorm1d(decoder_layers_size[i + 1]))
            nn.init.xavier_normal_(self.decoder_conv_layers[-1].weight)

        # self.h2_decoder_conv_layers = nn.ModuleList()
        # self.h2_decoder_bn_layers = nn.ModuleList()
        # for i in range(len(decoder_layers_size) - 1):
        #     self.h2_decoder_conv_layers.append(
        #         nn.Conv1d(decoder_layers_size[i], decoder_layers_size[i + 1], 1)
        #     )
        #     self.h2_decoder_bn_layers.append(nn.BatchNorm1d(decoder_layers_size[i + 1]))
        #     nn.init.xavier_normal_(self.h2_decoder_conv_layers[-1].weight)

    def forward(self, x, z_latent_code):
        """
        :param x: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        bs = x.shape[0]
        npts = x.shape[1]

        pointwise_feature = x.transpose(1, 2)
        for i in range(len(self.pointwise_conv_layers) - 1):
            pointwise_feature = self.pointwise_conv_layers[i](pointwise_feature)
            pointwise_feature = self.pointwise_bn_layers[i](pointwise_feature)
            pointwise_feature = self.activate_func(pointwise_feature)
        pointwise_feature = self.pointwise_bn_layers[-1](
            self.pointwise_conv_layers[-1](pointwise_feature)
        )

        global_feature = pointwise_feature.clone()
        for i in range(len(self.global_conv_layers) - 1):
            global_feature = self.global_conv_layers[i](global_feature)
            global_feature = self.global_bn_layers[i](global_feature)
            global_feature = self.activate_func(global_feature)
        global_feature = self.global_bn_layers[-1](
            self.global_conv_layers[-1](global_feature)
        )
        global_feature = torch.max(global_feature, 2, keepdim=True)[0]
        global_feature = global_feature.view(bs, self.global_feat_size)

        global_feature = torch.cat([global_feature, z_latent_code], dim=1)
        global_feature = global_feature.view(
            bs, self.global_feat_size + self.latent_size, 1
        ).repeat(1, 1, npts)
        pointwise_feature = torch.cat([pointwise_feature, global_feature], dim=1)
        # pointwise_feature_h2 = pointwise_feature.clone()
        for i in range(len(self.decoder_conv_layers) - 1):
            pointwise_feature = self.decoder_conv_layers[i](pointwise_feature)
            pointwise_feature = self.decoder_bn_layers[i](pointwise_feature)
            pointwise_feature = self.activate_func(pointwise_feature)
        pointwise_feature = self.decoder_bn_layers[-1](
            self.decoder_conv_layers[-1](pointwise_feature)
        )

        # for i in range(len(self.h2_decoder_conv_layers) - 1):
        #     pointwise_feature_h2 = self.h2_decoder_conv_layers[i](pointwise_feature_h2)
        #     pointwise_feature_h2 = self.h2_decoder_bn_layers[i](pointwise_feature_h2)
        #     pointwise_feature_h2 = self.activate_func(pointwise_feature_h2)
        # pointwise_feature_h2 = self.h2_decoder_bn_layers[-1](
        #     self.h2_decoder_conv_layers[-1](pointwise_feature_h2)
        # )

        ### pointwise_feature shape B x out_size x N
        # pointwise_feature = (
        #     self.sigmoid(pointwise_feature).view(bs, npts, -1).squeeze(-1)
        # )
        # Keep this without sigmoid, since we might do additional transforms
        # return pointwise_feature.view(bs, npts, -1).squeeze(-1)
        return pointwise_feature  # shape (bs, -1, npts)
        # output =  torch.cat((pointwise_feature, pointwise_feature_h2), dim=1).view(bs, npts, -1)
        # return output
        # return self.sigmoid(output)


class GcsCVAE(nn.Module):
    def __init__(
        self,
        latent_size=128,
        encoder_layers_size=[5, 64, 128, 512],
        decoder_global_feat_size=512,
        decoder_pointwise_layers_size=[3, 64, 64],
        decoder_global_layers_size=[64, 128, 512],
        decoder_decoder_layers_size=[64 + 512 + 128, 512, 64, 64, 1],
        num_coarse_pts=2048,
        uv_layers_size=None,
    ):
        # NOTE:
        # encoder_layers_size[0]  is 5 instead of 4 since our input is (obj_pc, obj_uv_coords)
        super(GcsCVAE, self).__init__()
        self.num_coarse = num_coarse_pts
        self.latent_size = latent_size

        self.cmap_encoder = PointNetCmapEncoder(layers_size=encoder_layers_size)
        self.cmap_decoder = PointNetCmapDecoder(
            latent_size=latent_size,
            global_feat_size=decoder_global_feat_size,
            pointwise_layers_size=decoder_pointwise_layers_size,
            global_layers_size=decoder_global_layers_size,
            decoder_layers_size=decoder_decoder_layers_size,
        )

        self.encoder_z_means = nn.Linear(encoder_layers_size[-1], latent_size)
        self.encoder_z_logvars = nn.Linear(encoder_layers_size[-1], latent_size)

        self.pred_uv = False
        if uv_layers_size:
            # uv_layers_size = [64, 64, 1]
            self.pred_uv = True
            num_layers = len(uv_layers_size) - 1

            self._pred_u_layers = nn.ModuleList()
            self._pred_v_layers = nn.ModuleList()
            for i in range(num_layers):
                curr_size, next_size = uv_layers_size[i], uv_layers_size[i + 1]

                self._pred_u_layers.append(nn.Conv1d(curr_size, next_size, 1))
                nn.init.xavier_normal_(self._pred_u_layers[-1].weight)
                self._pred_u_layers.append(nn.BatchNorm1d(next_size))

                self._pred_v_layers.append(nn.Conv1d(curr_size, next_size, 1))
                nn.init.xavier_normal_(self._pred_v_layers[-1].weight)
                self._pred_v_layers.append(nn.BatchNorm1d(next_size))

                if i < num_layers:
                    self._pred_u_layers.append(nn.ReLU())
                    self._pred_v_layers.append(nn.ReLU())

            self.pred_u_net = nn.Sequential(*self._pred_u_layers)
            self.pred_v_net = nn.Sequential(*self._pred_v_layers)

    def forward(self, obj_pts, gcs_gt):
        """
        :param obj_pts: B, N, 3
        :param gcs_gt: B, N, 2
        :return:
        """

        bs = obj_pts.shape[0]
        npts = obj_pts.shape[1]
        obj_cmap = torch.cat(
            (
                obj_pts,
                gcs_gt.unsqueeze(-1) if gcs_gt.ndim == 2 else gcs_gt,
            ),
            dim=-1,
        )
        means, logvars = self.forward_encoder(object_cmap=obj_cmap)
        z_latent_code = self.reparameterize(means=means, logvars=logvars)
        cmap_values = self.forward_decoder(obj_pts, z_latent_code).view(bs, npts, -1)
        return obj_pts, cmap_values, means, logvars, z_latent_code

    def predict(self, object_pts):
        """
        Test time prediction of contact maps from randomly sampled latent vectors on a given input

        Input:
         object_pts: (B, N, 3) tensor

        Returns:
         gcs_values: (B, N, 2) tensor of contact map values for each batched obj pc
        """
        bsize = object_pts.shape[0]
        z_samples = torch.randn(
            bsize, self.latent_size, device=object_pts.device
        ).float()
        return self.inference(object_pts, z_samples)

    def inference(self, object_pts, z_latent_code):
        """
        :param object_pts: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        cmap_values = self.forward_decoder(object_pts, z_latent_code)
        return cmap_values

    def reparameterize(self, means, logvars):
        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(std)
        return means + eps * std

    def forward_encoder(self, object_cmap):
        cmap_feat = self.cmap_encoder(object_cmap)
        means = self.encoder_z_means(cmap_feat)
        logvars = self.encoder_z_logvars(cmap_feat)
        return means, logvars

    def forward_decoder(self, object_pts, z_latent_code):
        """
        :param object_pts: B x N x 3
        :param z_latent_code: B x latent_size
        :return:
        """
        cmap_values = self.cmap_decoder(object_pts, z_latent_code)
        bs = cmap_values.shape[0]
        npts = cmap_values.shape[-1]
        if self.pred_uv:
            u = self.pred_u_net(cmap_values).view(bs, npts, -1)
            v = self.pred_v_net(cmap_values).view(bs, npts, -1)
            # u = torch.sigmoid(u)
            # v = torch.sigmoid(v)
            output = torch.cat([u, v], dim=-1).view(bs, npts, -1)
            return output
        else:
            return cmap_values.view(bs, npts, -1)

