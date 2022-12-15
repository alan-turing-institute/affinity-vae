import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from loss import AffinityLoss


class Encoder(nn.Module):
    def __init__(self, channel_init, latent_dims, depth, bottom_dim, pose=True, pose_dims=1):
        super(Encoder, self).__init__()
        c = channel_init
        self.pose = pose
        self.depth = depth

        self.conv_enc = nn.ModuleList()
        self.norm_enc = nn.ModuleList()
        prev_sh = 1
        for d in range(depth):
            sh = c * (d + 1)
            self.conv_enc.append(nn.Conv3d(in_channels=prev_sh, out_channels=sh, kernel_size=3, padding=1,
                                           stride=2))
            self.norm_enc.append(nn.BatchNorm3d(sh))
            prev_sh = sh

        if depth == 0:
            chf = 1
        else:
            chf = c * depth
        self.fc_mu = nn.Linear(in_features=chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2],
                               out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2], out_features=latent_dims)
        if pose:
            self.pose = nn.Linear(in_features=chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2],
                                  out_features=pose_dims)

    def forward(self, x):
        for d in range(self.depth):
            x = self.norm_enc[d](
                F.relu(self.conv_enc[d](x)))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, channel_init, latent_dims, depth, bottom_dim, pose=True, pose_dims=1):
        super(Decoder, self).__init__()
        self.c = channel_init
        self.depth = depth
        self.bottom_dim = bottom_dim
        self.pose = pose

        self.conv_dec = nn.ModuleList()
        self.norm_dec = nn.ModuleList()
        prev_sh = self.c * depth
        for d in range(depth, 0, -1):
            sh = self.c * (d - 1) if d != 1 else 1
            self.conv_dec.append(
                nn.ConvTranspose3d(in_channels=prev_sh, out_channels=sh, kernel_size=4, stride=2, padding=1)) # , output_padding=1))
            self.norm_dec.append(nn.BatchNorm3d(sh))
            prev_sh = sh
        # self.conv = nn.Conv3d(in_channels=prev_sh, out_channels=1, kernel_size=1)

        if depth == 0:
            self.chf = 1
        else:
            self.chf = self.c * depth
        if self.pose:
            self.fc = nn.Linear(in_features=pose_dims + latent_dims,
                                out_features=self.chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2])
        else:
            self.fc = nn.Linear(in_features=latent_dims,
                                out_features=self.chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2])

    def forward(self, x, pose):
        if self.pose:
            x = self.fc(torch.cat((x, pose), -1))
        else:
            x = self.fc(x)
        x = x.view(x.size(0), self.chf, self.bottom_dim[0], self.bottom_dim[1], self.bottom_dim[2])
        for d in range(self.depth - 1):
            x = self.norm_dec[d](F.relu(self.conv_dec[d](x)))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, channel_init, latent_dims, depth, input_size, lookup, gamma, pose=True, pose_dims=1):
        super(VariationalAutoencoder, self).__init__()
        assert all([int(x) == x for x in np.array(input_size) / (2 ** depth)]), \
            "Input size not compatible with --depth. Input must be divisible by {}.".format(2 ** depth)
        self.bottom_dim = tuple([int(i / (2 ** depth)) for i in input_size])
        self.pose = pose

        self.encoder = Encoder(channel_init, latent_dims, depth, self.bottom_dim, pose=pose, pose_dims=pose_dims)
        self.decoder = Decoder(channel_init, latent_dims, depth, self.bottom_dim, pose=pose, pose_dims=pose_dims)
        self.affinity_loss = AffinityLoss(lookup)

    def forward(self, x):
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
        latent = self.sample(latent_mu, latent_logvar)
        if self.pose:
            x_recon = self.decoder(latent, latent_pose)
        else:
            x_recon = self.decoder(latent, None)
        return x_recon, latent_mu, latent_logvar, latent, latent_pose

    def sample(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(self, recon_x, x, mu, logvar, beta,
             gamma=None, ids=None, lats=None,
             loss_fn='MSE', device=None, epoch=0):
        if lats is None:
            # chose mus or reparametrised latents for affinity calculation
            lats = mu
        sh = self.bottom_dim[0] * self.bottom_dim[1] * self.bottom_dim[2]
        if loss_fn == 'BCE':
            recon_loss = F.binary_cross_entropy(recon_x.view(-1, sh), x.view(-1, sh), reduction='mean')
        elif loss_fn == 'MSE':
            recon_loss = F.mse_loss(recon_x.view(-1, sh), x.view(-1, sh), reduction='mean')
        kldivergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        affin_loss = self.affinity_loss(ids, mu, device)

        return recon_loss + beta * kldivergence + gamma * affin_loss, recon_loss, kldivergence, affin_loss
