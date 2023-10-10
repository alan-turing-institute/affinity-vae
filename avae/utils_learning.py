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


def early_stopping(val_loss, patience, condition="total_loss"):
    """
    Early stopping function. Returns true if validation loss doesn't improve after a given patience.
    Depending on the condition, evaluation of loss improuvement can be done on
    the total loss, reconstruction loss, KL divergence or affinity loss.

    Parameters
    ----------
    val_loss: list
        List of validation losses.
    patience: int
        Number of previous epochs to use on the evaluation of loss improvement.
    condition: str
        Condition to evaluate loss improvement. Can be "total_loss", "reco_loss", "kldiv_loss" or "affinity_loss" or "all".
    """
    if condition not in [
        "total_loss",
        "reco_loss",
        "kldiv_loss",
        "affinity_loss",
        "all",
    ]:
        raise ValueError(
            "condition must be 'total_loss', 'reco_loss', 'kldiv_loss', 'affinity_loss' or 'all'"
        )

    stop = False
    if patience < len(val_loss):

        total_val_loss = [v[0] for v in val_loss][-patience:-1]
        val_loss_reco = [v[1] for v in val_loss][-patience:-1]
        val_loss_kl = [v[2] for v in val_loss][-patience:-1]
        val_loss_affinity = [v[3] for v in val_loss][-patience:-1]

        if total_val_loss.index(min(total_val_loss)) == 0 and (
            condition == "total_loss" or condition == "all"
        ):
            stop = True
        elif (condition == "reco_loss" or condition == "all") and (
            val_loss_reco.index(min(val_loss_reco)) == 0
        ):
            stop = True
        elif (condition == "kldiv_loss" or condition == "all") and (
            val_loss_kl.index(min(val_loss_kl)) == 0
        ):
            stop = True
        elif (condition == "affinity_loss" or condition == "all") and (
            val_loss_affinity.index(min(val_loss_affinity)) == 0
        ):
            stop = True

    return stop
