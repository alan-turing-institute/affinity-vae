import numpy as np
import pandas as pd
import torch
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier

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
        print(
            "\nWARNING: no GPU available, running on CPU instead.\n",
            flush=True,
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
        print(
            "Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f"
            % (e + 1, epochs, b + 1, batches, *history_loss, beta[e]),
            end="\r",
            flush=True,
        )

    # backwards
    if optimizer is not None:
        history_loss[0].backward()
        optimizer.step()
        optimizer.zero_grad()

    return x, x_hat, lat_mu, lat_logvar, lat, lat_pose, history


def add_meta(
    meta_df, batch_meta, x_hat, latent_mu, lat_pose, latent_logvar, mode="trn"
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
    meta["image"] += vis.format(x_hat)
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


def accuracy(x_train, y_train, x_val, y_val):
    """Computes the accuracy using a KNN classifier.

    Parameters
    ----------
    x_train: np.array
        Training data.
    y_train: np.array
        Training labels.
    x_val: np.array
        Validation data.
    y_val: np.array
        Validation labels.

    Returns
    -------
    train_acc: float
        Training accuracy.
    val_acc: float
        Validation accuracy.
    y_pred_train: np.array
        Predicted training labels.
    y_pred_val: np.array
        Predicted validation labels.

    """
    labs = np.unique(np.concatenate((y_train, y_val)))
    le = preprocessing.LabelEncoder()
    le.fit(labs)

    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    neigh = KNeighborsClassifier(n_neighbors=20)
    neigh.fit(x_train, y_train)
    y_pred_train = neigh.predict(x_train)
    y_pred_val = neigh.predict(x_val)
    train_acc = metrics.accuracy_score(y_train, y_pred_train)
    val_acc = metrics.accuracy_score(y_val, y_pred_val)

    y_pred_train = le.inverse_transform(y_pred_train)
    y_pred_val = le.inverse_transform(y_pred_val)

    return train_acc, val_acc, y_pred_train, y_pred_val
