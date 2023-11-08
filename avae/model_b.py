import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SpatialDims


def set_layer_dim(ndim):
    global CONV
    global TCONV
    global BNORM
    if ndim == SpatialDims.TWO:
        CONV = nn.Conv2d
        TCONV = nn.ConvTranspose2d
        BNORM = nn.BatchNorm2d
    elif ndim == SpatialDims.THREE:
        CONV = nn.Conv3d
        TCONV = nn.ConvTranspose3d
        BNORM = nn.BatchNorm3d
    else:
        logging.error("Data must be 2D or 3D.")
        exit(1)


class Encoder(nn.Module):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    bottom_dim: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size after downsampling for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(self, capacity, depth, bottom_dim, latent_dims, pose_dims=0):
        super(Encoder, self).__init__()
        c = capacity
        self.depth = depth
        self.pose = not (pose_dims == 0)

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = nn.ModuleList()
        self.norm_enc = nn.ModuleList()
        prev_sh = 1
        for d in range(depth):
            sh = c * (d + 1)
            self.conv_enc.append(
                CONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.norm_enc.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        chf = 1 if depth == 0 else c * depth  # allow for no conv layers
        self.fc_mu = nn.Linear(
            in_features=chf * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        self.fc_logvar = nn.Linear(
            in_features=chf * np.prod(bottom_dim),
            out_features=latent_dims,
        )
        if self.pose:
            self.fc_pose = nn.Linear(
                in_features=chf * np.prod(bottom_dim),
                out_features=pose_dims,
            )

    def forward(self, x):
        """Encoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        x_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.
        """
        for d in range(self.depth):
            x = self.norm_enc[d](F.relu(self.conv_enc[d](x)))
        x = x.view(x.size(0), -1)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        if self.pose:
            x_pose = self.fc_pose(x)
            return x_mu, x_logvar, x_pose
        else:
            return x_mu, x_logvar


class Decoder(nn.Module):
    """Affinity decoder. Includes optional pose component merge.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    bottom_dim: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size after downsampling for each image
        dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(self, capacity, depth, bottom_dim, latent_dims, pose_dims=0):
        super(Decoder, self).__init__()
        self.c = capacity
        self.depth = depth
        self.bottom_dim = bottom_dim
        self.pose = not (pose_dims == 0)

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = nn.ModuleList()
        self.norm_dec = nn.ModuleList()
        prev_sh = self.c * depth
        for d in range(depth, 0, -1):
            sh = self.c * (d - 1) if d != 1 else 1
            self.conv_dec.append(
                TCONV(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.norm_dec.append(BNORM(sh))
            prev_sh = sh

        # define fully connected layers
        self.chf = (
            1 if depth == 0 else self.c * depth
        )  # allow for no convolutions
        if self.pose:
            self.fc = nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.chf * np.prod(bottom_dim),
            )
        else:
            self.fc = nn.Linear(
                in_features=latent_dims,
                out_features=self.chf * np.prod(bottom_dim),
            )

    def forward(self, x, x_pose):
        """Decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch and
            'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Mini-batch of outputs representing pose capturing the within-class
            variance, where N stands for the number
            of samples in the mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        """
        if self.pose:
            x = self.fc(torch.cat((x, x_pose), -1))
        else:
            x = self.fc(x)
        x = x.view(x.size(0), self.chf, *self.bottom_dim)
        for d in range(self.depth - 1):
            x = self.norm_dec[d](F.relu(self.conv_dec[d](x)))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


class AffinityVAE(nn.Module):
    """Affinity regularised Variational Autoencoder with an optional
    within-class variance encoding pose component.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each
        depth.
    depth : int
        The depth of the network - number of downsampling layers.
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the input for each dimension  X, Y and
        Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self,
        capacity,
        depth,
        input_size,
        latent_dims,
        pose_dims=0,
        device=None,
    ):
        super(AffinityVAE, self).__init__()
        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), (
            "Input size not compatible with --depth. Input must be divisible "
            "by {}.".format(2**depth)
        )
        set_layer_dim(len(input_size))
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])
        self.pose = not (pose_dims == 0)
        self.device = device

        self.encoder = Encoder(
            capacity,
            depth,
            self.bottom_dim,
            latent_dims,
            pose_dims=pose_dims,
        )
        self.decoder = Decoder(
            capacity,
            depth,
            self.bottom_dim,
            latent_dims,
            pose_dims=pose_dims,
        )

    def forward(self, x):
        """AffinityVAE forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_recon : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in
            the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        latent_mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N
            stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent
            dimensions.
        latent_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the
            variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of
            latent dimensions.
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        latent_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of encoder outputs
            representing pose capturing the within-class
            variance, where N stands for the number of samples in the
            mini-batch and 'pose_dims' defines the number of
            pose dimensions.

        """
        # encode
        if self.pose:
            latent_mu, latent_logvar, latent_pose = self.encoder(x)
        else:
            latent_mu, latent_logvar = self.encoder(x)
            latent_pose = None
        # reparametrise
        latent = self.sample(latent_mu, latent_logvar)
        # decode
        x_recon = self.decoder(latent, latent_pose)  # pose set to None if pd=0
        if self.pose:
            from scipy.ndimage import rotate

            # import matplotlib.pyplot as plt
            ims = x_recon.cpu().detach().numpy()
            theta = latent_pose.cpu().detach().numpy()
            for i, im in enumerate(ims):
                # print(theta[i][0]*360)
                # plt.imshow(ims[i][0])
                # plt.show()
                ims[i] = rotate(
                    im[0],
                    angle=theta[i][0] * 360,
                    axes=(-2, -1),
                    order=1,
                    reshape=False,
                )
                # plt.imshow(ims[i][0])
                # plt.show()
            x_recon = torch.Tensor(ims)
            x_recon.to(self.device)
        return x_recon, latent_mu, latent_logvar, latent, latent_pose

    def sample(self, mu, logvar):
        """Reparametrisation trick.

        Parameters
        ----------
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands
            for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance,
            where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent
            dimensions.

        Returns
        -------
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for
            the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
