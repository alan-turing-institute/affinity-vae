import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import AffinityLoss


class Encoder(nn.Module):
    """Affinity encoder. Includes optional pose component in the architecture.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each depth.
    depth : int
        The depth of the network - number of downsampling layers.
    bottom_dim: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size after downsampling for each image dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self, capacity, depth, bottom_dim, latent_dims, pose=True, pose_dims=1
    ):
        super(Encoder, self).__init__()
        c = capacity
        self.pose = pose
        self.depth = depth

        # iteratively define convolution and batch normalisation layers
        self.conv_enc = nn.ModuleList()
        self.norm_enc = nn.ModuleList()
        prev_sh = 1
        for d in range(depth):
            sh = c * (d + 1)
            self.conv_enc.append(
                nn.Conv3d(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.norm_enc.append(nn.BatchNorm3d(sh))
            prev_sh = sh

        # define fully connected layers
        chf = 1 if depth == 0 else c * depth  # allow for no conv layers
        self.fc_mu = nn.Linear(
            in_features=chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2],
            out_features=latent_dims,
        )
        self.fc_logvar = nn.Linear(
            in_features=chf * bottom_dim[0] * bottom_dim[1] * bottom_dim[2],
            out_features=latent_dims,
        )
        if pose:
            self.fc_pose = nn.Linear(
                in_features=chf
                * bottom_dim[0]
                * bottom_dim[1]
                * bottom_dim[2],
                out_features=pose_dims,
            )

    def forward(self, x):
        """Encoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        x_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance, where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of outputs representing pose capturing the within-class
            variance, where N stands for the number of samples in the mini-batch and 'pose_dims' defines the number of
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
        The capacity of the network - initial number of nodes doubled at each depth.
    depth : int
        The depth of the network - number of downsampling layers.
    bottom_dim: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size after downsampling for each image dimension X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    pose: bool
        Determines whether pose component is on or off.
    pose_dims : int
        Number of bottleneck pose dimensions.
    """

    def __init__(
        self, capacity, depth, bottom_dim, latent_dims, pose=True, pose_dims=1
    ):
        super(Decoder, self).__init__()
        self.c = capacity
        self.depth = depth
        self.bottom_dim = bottom_dim
        self.pose = pose

        #  iteratively define deconvolution and batch normalisation layers
        self.conv_dec = nn.ModuleList()
        self.norm_dec = nn.ModuleList()
        prev_sh = self.c * depth
        for d in range(depth, 0, -1):
            sh = self.c * (d - 1) if d != 1 else 1
            self.conv_dec.append(
                nn.ConvTranspose3d(
                    in_channels=prev_sh,
                    out_channels=sh,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            self.norm_dec.append(nn.BatchNorm3d(sh))
            prev_sh = sh

        # define fully connected layers
        self.chf = (
            1 if depth == 0 else self.c * depth
        )  # allow for no convolutions
        if self.pose:
            self.fc = nn.Linear(
                in_features=pose_dims + latent_dims,
                out_features=self.chf
                * bottom_dim[0]
                * bottom_dim[1]
                * bottom_dim[2],
            )
        else:
            self.fc = nn.Linear(
                in_features=latent_dims,
                out_features=self.chf
                * bottom_dim[0]
                * bottom_dim[1]
                * bottom_dim[2],
            )

    def forward(self, x, x_pose):
        """Decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for the number of samples in the mini-batch and
            'latent_dims' defines the number of latent dimensions.
        x_pose : torch.Tensor (N, pose_dims)
            Mini-batch of outputs representing pose capturing the within-class variance, where N stands for the number
            of samples in the mini-batch and 'pose_dims' defines the number of pose dimensions.

        Returns
        -------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        """
        if self.pose:
            x = self.fc(torch.cat((x, x_pose), -1))
        else:
            x = self.fc(x)
        x = x.view(
            x.size(0),
            self.chf,
            self.bottom_dim[0],
            self.bottom_dim[1],
            self.bottom_dim[2],
        )
        for d in range(self.depth - 1):
            x = self.norm_dec[d](F.relu(self.conv_dec[d](x)))
        x = torch.sigmoid(self.conv_dec[-1](x))
        return x


class AffinityVAE(nn.Module):
    """Affinity regularised Variational Autoencoder with an optional within-class variance encoding pose component.

    Parameters
    ----------
    capacity : int
        The capacity of the network - initial number of nodes doubled at each depth.
    depth : int
        The depth of the network - number of downsampling layers.
    input_size: tuple (X, Y) or tuple (X, Y, Z)
        Tuple representing the size of the input for each dimension  X, Y and Z.
    latent_dims: int
        Number of bottleneck latent dimensions.
    lookup : np.ndarray [M, M]
        A square symmetric matrix where each column and row is the index of an object class from the training set,
        consisting of M different classes.
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
        lookup,
        pose=True,
        pose_dims=1,
    ):
        super(AffinityVAE, self).__init__()
        assert all(
            [int(x) == x for x in np.array(input_size) / (2**depth)]
        ), "Input size not compatible with --depth. Input must be divisible by {}.".format(
            2**depth
        )
        self.bottom_dim = tuple([int(i / (2**depth)) for i in input_size])
        self.pose = pose

        self.encoder = Encoder(
            capacity,
            depth,
            self.bottom_dim,
            latent_dims,
            pose=pose,
            pose_dims=pose_dims,
        )
        self.decoder = Decoder(
            capacity,
            depth,
            self.bottom_dim,
            latent_dims,
            pose=pose,
            pose_dims=pose_dims,
        )
        self.affinity_loss = AffinityLoss(lookup)

    def forward(self, x):
        """AffinityVAE forward pass.

        Parameters
        ----------
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.

        Returns
        -------
        x_recon : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        latent_mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent dimensions.
        latent_logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of latent dimensions.
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        latent_pose : torch.Tensor (N, pose_dims)
            Optional return if pose is True. Mini-batch of encoder outputs representing pose capturing the within-class
            variance, where N stands for the number of samples in the mini-batch and 'pose_dims' defines the number of
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
        if self.pose:
            x_recon = self.decoder(latent, latent_pose)
        else:
            x_recon = self.decoder(latent, None)
        return x_recon, latent_mu, latent_logvar, latent, latent_pose

    def sample(self, mu, logvar):
        """Reparametrisation trick.

        Parameters
        ----------
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent means, where N stands for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of outputs representing latent log of the variance, where N stands for the number of samples in
            the mini-batch and 'latent_dims' defines the number of latent dimensions.

        Returns
        -------
        latent : torch.Tensor (N, latent_dims)
            Mini-batch of reparametrised encoder outputs, where N stands for the number of samples in the mini-batch
            and 'latent_dims' defines the number of latent dimensions.
        """
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss(
        self,
        recon_x,
        x,
        mu,
        logvar,
        beta,
        gamma=None,
        ids=None,
        loss_fn="MSE",
        device=None,
    ):
        """AffinityVAE loss consisting of reconstruction loss, beta parametrised latent regularisation loss and
        gamma parametrised affinity regularisation loss. Reconstruction loss should be Mean Squared Error for real
        valued data and Binary Cross-Entropy for binary data. Latent regularisation loss is KL Divergence. Affinity
        regularisation is defined in loss.py.

        Parameters
        ----------
        recon_x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        x : torch.Tensor (N, CH, Z, Y, X)
            Mini-batch of inputs, where N stands for the number of samples in the mini-batch, CH stands for number of
            channels and X, Y, Z define input dimensions.
        mu : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent means, where N stands for the number of samples in the
            mini-batch and 'latent_dims' defines the number of latent dimensions.
        logvar : torch.Tensor (N, latent_dims)
            Mini-batch of encoder outputs representing latent log of the variance, where N stands for the number of
            samples in the mini-batch and 'latent_dims' defines the number of latent dimensions.
        beta: float
            Beta parameter defining weight on latent regularisation term.
        gamma : float
            Gamma parameter defining weight on affinity regularisation term. If gamma is None, affinity loss
            is not computed and does not count towards the total loss.
        ids : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing the identity of the object's class as
            an index. These indices should correspond to the rows and columns of the `lookup` table.
        loss_fn : 'MSE' or 'BCE'
            Function used for reconstruction loss. BCE uses Binary Cross-Entropy for binary data and MSE uses Mean
            Squared Error for real-valued data.
        device : torch.device
            Device used to calculate the loss.

        Returns
        -------
        total_loss : torch.Tensor
            Combined reconstruction, latent regularisaton and affinity loss.
        recon_loss : torch.Tensor
            Reconstruction loss.
        kldivergence : torch.Tensor
            Non-weighted KL Divergence loss.
        affin_loss : torch.Tensor
            Non-weighted affinity loss.
        """

        # recon loss
        sh = self.bottom_dim[0] * self.bottom_dim[1] * self.bottom_dim[2]
        if loss_fn == "BCE":
            recon_loss = F.binary_cross_entropy(
                recon_x.view(-1, sh), x.view(-1, sh), reduction="mean"
            )
        elif loss_fn == "MSE":
            recon_loss = F.mse_loss(
                recon_x.view(-1, sh), x.view(-1, sh), reduction="mean"
            )
        else:
            raise RuntimeError(
                "AffinityVAE loss requires 'BCE' or 'MSE' for 'loss_fn' parameter."
            )

        # kldiv loss
        kldivergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # affinity loss
        if gamma is not None:
            affin_loss = self.affinity_loss(ids, mu, device)
        else:
            affin_loss, gamma = 0, 0

        total_loss = recon_loss + beta * kldivergence + gamma * affin_loss

        return total_loss, recon_loss, kldivergence, affin_loss
