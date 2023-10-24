import logging

import numpy as np
import pandas as pd
import torch

from . import vis


def set_device(gpu):
    """Set the torch device to use for training and inference.

    Parameters
    ----------
    gpu: bool
        If True, the model will be trained on GPU.

    Returns
    -------
    device: torch.device

    """
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        logging.warning(
            "\n\nWARNING: no GPU available, running on CPU instead.\n"
        )
    return device


def dims_after_pooling(start: int, n_pools: int) -> int:
    """Calculate the size of a layer after n pooling ops.

    Parameters
    ----------
    start: int
        The size of the layer before pooling.
    n_pools: int
        The number of pooling operations.

    Returns
    -------
    int
        The size of the layer after pooling.


    """
    return start // (2**n_pools)


def pass_batch(
    device,
    vae,
    batch,
    b,
    batches,
    e=None,
    epochs=None,
    history=[],
    loss=None,
    optimizer=None,
    beta=None,
):
    """Passes a batch through the affinity VAE model epoch and computes the loss.

    Parameters
    ----------
    device: torch.device
        Device to use for training.
    vae: torch.nn.Module
        Affinity VAE model class.
    batch: list
        List of batches with data and labels.
    b: int
        Batch number.
    batches: int
        Total number of batches.
    e: int
        Epoch number.
    epochs: int
        Total number of epochs.
    history: list
        List of training losses.
    loss: avae.loss.AVAELoss
        Loss function class.
    optimizer: torch.optim
        Optimizer.
    beta: float
        Beta parameter for affinity-VAE.

    Returns
    -------
    x: torch.Tensor
        Input data.
    x_hat: torch.Tensor
        Reconstructed data.
    lat_mu: torch.Tensor
        Latent mean.
    lat_logvar: torch.Tensor
        Latent log variance.
    lat: torch.Tensor
        Latent representation.
    lat_pose: torch.Tensor
        Latent pose.
    history: list
        List of training losses.


    """
    if bool(history == []) ^ bool(loss is None):
        raise RuntimeError(
            "When validating, both 'loss' and 'history' parameters must be "
            "present in 'pass_batch' function."
        )
    if bool(e is None) ^ bool(epochs is None):
        raise RuntimeError(
            "Function 'pass_batch' expects both 'e' and 'epoch' parameters."
        )
    if e is None and epochs is None:
        e = 1
        epochs = 1

    # to device
    x = batch[0]
    x = x.to(device)
    aff = batch[2]
    aff = aff.to(device)

    # forward
    x = x.to(torch.float32)
    x_hat, lat_mu, lat_logvar, lat, lat_pose = vae(x)
    if loss is not None:
        history_loss = loss(x, x_hat, lat_mu, lat_logvar, e, batch_aff=aff)

        if beta is None:
            raise RuntimeError(
                "Please pass beta value to pass_batch function."
            )

        # record loss
        for i in range(len(history[-1])):
            history[-1][i] += history_loss[i].item()
        logging.debug(
            "Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f"
            % (e + 1, epochs, b + 1, batches, *history_loss, beta[e])
        )

    # backwards
    if optimizer is not None:
        history_loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()

    return x, x_hat, lat_mu, lat_logvar, lat, lat_pose, history


def add_meta(
    data_dim,
    meta_df,
    batch_meta,
    x_hat,
    latent_mu,
    lat_pose,
    latent_logvar,
    mode="trn",
):
    """
    Created meta data about data and training.

    Parameters
    ----------
    meta_df: pd.DataFrame
        Dataframe containing meta data, to which new data is added.
    batch_meta: dict
        Meta data about the batch.
    x_hat: torch.Tensor
        Reconstructed data.
    latent_mu: torch.Tensor
        Latent mean.
    lat_pose: torch.Tensor
        Latent pose.
    lat_logvar: torch.Tensor
        Latent logvar.
    mode: str
        Data category on training (either 'trn', 'val' or 'test').

    Returns
    -------
    meta_df: pd.DataFrame
        Dataframe containing meta data.

    """
    meta = pd.DataFrame(batch_meta)

    meta["mode"] = mode
    meta["image"] += vis.format(x_hat, data_dim)
    for d in range(latent_mu.shape[-1]):
        meta[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
    for d in range(latent_logvar.shape[-1]):
        lat_std = np.exp(0.5 * latent_logvar[:, d].cpu().detach().numpy())
        meta[f"std-{d}"] = np.array(lat_std)
    if lat_pose is not None:
        for d in range(lat_pose.shape[-1]):
            meta[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
    meta_df = pd.concat(
        [meta_df, meta], ignore_index=False
    )  # ignore index doesn't overwrite
    return meta_df


class EarlyStopping:
    """
    Early stopping class. Returns true if validation loss doesn't improve after a given patience.
    Depending on the condition, evaluation of loss improuvement can be done on
    the total loss, reconstruction loss, KL divergence or affinity loss.

    Parameters
    ----------
    trigger: str
        Condition to evaluate loss improvement. Can be "total_loss", "reco_loss", "kldiv_loss" or "affinity_loss" or "all".
    patience: int
        Number of previous epochs to use on the evaluation of loss improvement.
    max_delta: float
        Minimum change in loss to be considered as an improvement (ratio).
    max_divergence: float
        Max change in validation loss with respect to the trainin loss (ratio).
    min_epochs: int
        Minimum number of epochs to run before early stopping can be triggered.

    """

    def __init__(
        self,
        loss_type: str,
        patience: int,
        max_delta: float,
        max_divergence: float,
        min_epochs: int,
    ):
        if loss_type not in [
            "total_loss",
            "reco_loss",
            "kldiv_loss",
            "affinity_loss",
            "all",
        ]:
            raise ValueError(
                "early stopping trigger must be 'total_loss', 'reco_loss', 'kldiv_loss', 'affinity_loss' or 'all'"
            )
        self.trigger = loss_type
        self.patience = patience
        self.max_delta = max_delta
        self.max_divergence = max_divergence
        self.min_epochs = min_epochs
        self.stop = False

    def early_stop(self, val_loss, train_loss):
        """
        Early stopping function. Returns true if validation loss doesn't improve after a given patience.
        Depending on the condition, evaluation of loss improuvement can be done on
        the total loss, reconstruction loss, KL divergence or affinity loss.

        Parameters
        ----------
        val_loss: list
            List of validation losses.
        train_loss: list
            List of validation losses.
        """

        if self.patience < len(val_loss) and len(val_loss) > self.min_epochs:

            total_val_loss = [v[0] for v in val_loss][-self.patience :]
            val_loss_reco = [v[1] for v in val_loss][-self.patience :]
            val_loss_kl = [v[2] for v in val_loss][-self.patience :]
            val_loss_affinity = [v[3] for v in val_loss][-self.patience :]

            total_train_loss = [v[0] for v in train_loss][-self.patience :]
            val_train_reco = [v[1] for v in train_loss][-self.patience :]
            val_train_kl = [v[2] for v in train_loss][-self.patience :]
            val_train_affinity = [v[3] for v in train_loss][-self.patience :]

            if self.trigger == "total_loss":

                if self.__evaluate_loss(total_val_loss, total_train_loss):
                    logging.info('Early stopping triggered on "total_loss"')
                    self.stop = True

            elif self.trigger == "reco_loss" or self.trigger == "all":

                if self.__evaluate_loss(val_loss_reco, val_train_reco):
                    logging.info('Early stopping triggered on "reco_loss"')
                    self.stop = True

            elif self.trigger == "kldiv_loss" or self.trigger == "all":
                if self.__evaluate_loss(val_loss_kl, val_train_kl):
                    logging.info('Early stopping triggered on "kldiv_loss"')
                    self.stop = True

            elif self.trigger == "affinity_loss" or self.trigger == "all":
                if self.__evaluate_loss(val_loss_affinity, val_train_affinity):
                    logging.info('Early stopping triggered on "affinity_loss"')
                    self.stop = True

        return self.stop

    def __evaluate_loss(self, valid_loss, train_loss):

        stop = False
        if valid_loss.index(min(valid_loss)) == 0:
            logging.info(
                "Early stopping triggered: validation loss is increasing"
            )
            stop = True

        if max(valid_loss) < min(valid_loss) + self.max_delta * min(
            valid_loss
        ):
            logging.info(
                "Early stopping triggered: validation loss is not improving. Fluctuation is within {:.2f}% of minimum validation loss.".format(
                    (max(valid_loss) - min(valid_loss)) / min(valid_loss) * 100
                )
            )
            stop = True

        diff_loses = (np.array(valid_loss) - np.array(train_loss)).tolist()
        if diff_loses.index(min(diff_loses)) == 0 and max(valid_loss) > max(
            train_loss
        ) + self.max_divergence * max(train_loss):
            logging.info(
                "Early stopping triggered: validation loss is diverging from training loss"
            )

            stop = True

        return stop
