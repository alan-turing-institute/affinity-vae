import datetime
import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier

from . import config, vis
from .cyc_annealing import cyc_annealing
from .data import load_data
from .loss import AVAELoss
from .model_a import AffinityVAE as AffinityVAE_A
from .model_b import AffinityVAE as AffinityVAE_B
from .utils import set_device


def train(
    datapath,
    lim,
    splt,
    batch_s,
    no_val_drop,
    affinity,
    classes,
    collect_meta,
    epochs,
    channels,
    depth,
    lat_dims,
    pose_dims,
    learning,
    beta_min,
    beta_max,
    beta_cycle,
    beta_ratio,
    cyc_method_beta,
    gamma_min,
    gamma_max,
    gamma_cycle,
    gamma_ratio,
    cyc_method_gamma,
    recon_fn,
    use_gpu,
    model,
):
    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    datapath: str
        Path to the data directory.
    lim: int
        Limit the number of samples to load.
    splt: int
        Percentage of data to be used for validation.
    batch_s: int
        Batch size.
    no_val_drop: bool
        If True, the last batch of validation data will not be dropped if it is smaller than batch size.
    affinity: str
        Path to the affinity matrix.
    classes: list
        List of classes to be selected from the data for the training and validation set.
    collect_meta: bool
        If True, the meta data for visualisation will be collected and returned.
    epochs: int
        Number of epochs to train the model.
    channels: int
        Number of channels in the input data.
    depth: int
        Depth of the model.
    lat_dims: int
        Number of latent dimensions.
    pose_dims: int
        Number of pose dimensions.
    learning: float
        Learning rate.
    beta_min: float
        Minimum value of beta.
    beta_max: float
        Maximum value of beta.
    beta_cycle: int
        Number of epochs for beta to cycle.
    beta_ratio: float
        Ratio of beta to gamma.
    cyc_method_beta: str
        Method of beta cycle.
    gamma_min: float
        Minimum value of gamma.
    gamma_max: float
        Maximum value of gamma.
    gamma_cycle: int
        Number of epochs for gamma to cycle.
    gamma_ratio: float
        Ratio of gamma to beta.
    cyc_method_gamma: str
        Method of gamma cycle.
    recon_fn: str
        Reconstruction loss function.
    use_gpu: bool
        If True, the model will be trained on GPU.
    model: str
        Type of model to train. Can be a or b.

    """
    torch.manual_seed(42)
    timestamp = str(datetime.datetime.now().strftime("%Y%m%d_%H:%M:%S"))

    # ############################### DATA ###############################
    trains, vals, tests, lookup = load_data(
        datapath,
        lim=lim,
        splt=splt,
        batch_s=batch_s,
        no_val_drop=no_val_drop,
        collect_meta=collect_meta,
        eval=False,
        affinity=affinity,
        classes=classes,
    )
    dshape = list(trains)[0][0].shape[-3:]
    pose = not (pose_dims == 0)

    # ############################### MODEL ###############################
    device = set_device(use_gpu)

    if model == "a":
        affinityVAE = AffinityVAE_A
    elif model == "b":
        affinityVAE = AffinityVAE_B
    else:
        raise ValueError("Invalid model type", model, "must be a or b")

    vae = affinityVAE(
        channels,
        depth,
        dshape,
        lat_dims,
        pose_dims=pose_dims,
    )
    vae.to(device)

    optimizer = torch.optim.Adam(
        params=vae.parameters(), lr=learning  # , weight_decay=1e-5
    )

    if beta_max == 0 and cyc_method_beta != "flat":
        raise RuntimeError(
            "The maximum value for beta is set to 0, it is not possible to"
            "oscillate between a maximum and minimum. Please choose the flat method for"
            "cyc_method_beta"
        )
    beta_arr = cyc_annealing(
        epochs,
        cyc_method_beta,
        start=beta_min,
        stop=beta_max,
        n_cycle=beta_cycle,
        ratio=beta_ratio,
    ).var

    if gamma_max == 0 and cyc_method_gamma != "flat":
        raise RuntimeError(
            "The maximum value for gamma is set to 0, it is not possible to"
            "oscillate between a maximum and minimum. Please choose the flat method for"
            "cyc_method_gamma"
        )
    gamma_arr = cyc_annealing(
        epochs,
        cyc_method_gamma,
        start=gamma_min,
        stop=gamma_max,
        n_cycle=gamma_cycle,
        ratio=gamma_ratio,
    ).var

    if config.VIS_CYC:
        vis.plot_cyc_variable(beta_arr, "beta")
        vis.plot_cyc_variable(gamma_arr, "gamma")

    loss = AVAELoss(
        device,
        beta_arr,
        gamma=gamma_arr,
        lookup_aff=lookup,
        recon_fn=recon_fn,
    )

    t_history = []
    v_history = []

    print(
        "Epoch: [0/%d] | Batch: [0/%d] | Loss: -- | Recon: -- | "
        "KLdiv: -- | Affin: -- | Beta: --" % (epochs, len(trains)),
        end="\r",
        flush=True,
    )

    # ########################## TRAINING LOOP ################################
    for epoch in range(epochs):

        if collect_meta:
            meta_df = pd.DataFrame()

        # populate loss with new epoch
        t_history.append(np.zeros(4))
        v_history.append(np.zeros(4))

        # create holders for latent spaces and labels
        x_train = []  # 0 x lat_dims
        y_train = []  # 0 x 1
        c_train = []
        x_val = []
        y_val = []
        c_val = []
        x_test = []
        c_test = []
        if pose:
            p_train = []  # 0 x pose_dims
            p_val = []
            p_test = []

        # ########################## TRAINING #################################
        vae.train()
        for b, batch in enumerate(trains):

            x, x_hat, lat_mu, lat_logvar, lat, lat_pos, t_history = pass_batch(
                device,
                vae,
                batch,
                b,
                len(trains),
                epoch,
                epochs,
                loss=loss,
                history=t_history,
                optimizer=optimizer,
                beta=beta_arr,
            )
            x_train.extend(lat_mu.cpu().detach().numpy())  # store latents
            y_train.extend(batch[1])
            c_train.extend(lat_logvar.cpu().detach().numpy())
            if pose:
                p_train.extend(lat_pos.cpu().detach().numpy())

            if collect_meta:  # store meta for plots
                meta_df = add_meta(
                    meta_df,
                    batch[-1],
                    x_hat,
                    lat_mu,
                    lat_pos,
                    lat_logvar,
                    mode="trn",
                )

        t_history[-1] /= len(trains)
        print(
            "Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f"
            % (
                epoch + 1,
                epochs,
                b + 1,
                len(trains),
                *t_history[-1],
                beta_arr[epoch],
            ),
            flush=True,
        )

        # ########################## VAL ######################################
        vae.eval()
        for b, batch in enumerate(vals):
            v, v_hat, v_mu, v_logvar, vlat, vlat_pos, v_history = pass_batch(
                device,
                vae,
                batch,
                b,
                len(vals),
                epoch,
                epochs,
                loss=loss,
                history=v_history,
                beta=beta_arr,
            )
            x_val.extend(v_mu.cpu().detach().numpy())  # store latents
            y_val.extend(batch[1])
            c_val.extend(v_logvar.cpu().detach().numpy())
            if pose:
                p_val.extend(vlat_pos.cpu().detach().numpy())

            if collect_meta:  # store meta for plots
                meta_df = add_meta(
                    meta_df,
                    batch[-1],
                    v_hat,
                    v_mu,
                    vlat_pos,
                    v_logvar,
                    mode="val",
                )

        v_history[-1] /= len(vals)
        print(
            "Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f"
            % (
                epoch + 1,
                epochs,
                b + 1,
                len(vals),
                *v_history[-1],
                beta_arr[epoch],
            ),
            flush=True,
        )

        # ########################## TEST #####################################
        if (epoch + 1) % config.FREQ_EVAL == 0:
            for b, batch in enumerate(tests):  # tests empty if no 'test' dir
                t, t_hat, t_mu, t_logvar, tlat, tlat_pose, _ = pass_batch(
                    device, vae, batch, b, len(tests), epoch, epochs
                )
                x_test.extend(t_mu.cpu().detach().numpy())  # store latents
                c_test.extend(t_logvar.cpu().detach().nupmy())
                if pose:
                    p_test.extend(tlat_pose.cpu().detach().numpy())

                if collect_meta:  # store meta for plots
                    meta_df = add_meta(
                        meta_df,
                        batch[-1],
                        t_hat,
                        t_mu,
                        tlat_pose,
                        t_logvar,
                        mode="tst",
                    )

        # ########################## SAVE STATE ###############################
        if (epoch + 1) % config.FREQ_STA == 0:
            if not os.path.exists("states"):
                os.mkdir("states")
            mname = (
                "avae_"
                + str(timestamp)
                + "_E"
                + str(epoch)
                + "_"
                + str(lat_dims)
                + "_"
                + str(pose_dims)
                + ".pt"
            )
            torch.save(vae, os.path.join("states", mname))
            if collect_meta:
                meta_df.to_pickle(
                    os.path.join("states", "meta_" + timestamp + ".pkl")
                )

        # ########################## VISUALISE ################################

        # visualise accuracy: confusion and F1 scores
        if config.VIS_ACC and (epoch + 1) % config.FREQ_ACC == 0:
            train_acc, val_acc, ypred_train, ypred_val = accuracy(
                x_train, y_train, x_val, y_val
            )
            print(
                "\n------------------->>> Accuracy: Train: %f | Val: %f\n"
                % (train_acc, val_acc),
                flush=True,
            )
            vis.accuracy_plot(y_train, ypred_train, y_val, ypred_val, classes)

        # visualise loss
        if config.VIS_LOS and epoch > 0:
            p = [
                len(trains),
                depth,
                channels,
                lat_dims,
                learning,
                beta_arr[epoch],
                gamma_arr[epoch],
            ]
            vis.loss_plot(epoch + 1, t_history, v_history, p=p)

        # visualise reconstructions - last batch
        if config.VIS_REC and (epoch + 1) % config.FREQ_REC == 0:
            vis.recon_plot(x, x_hat, name="trn")
            vis.recon_plot(v, v_hat, name="val")

        if config.VIS_CON and (epoch + 1) % config.FREQ_CON == 0:
            vis.confidence_plot(x_train, y_train, c_train, suffix="trn")
            vis.confidence_plot(x_val, y_val, c_val, suffix="val")

        # visualise embeddings
        if config.VIS_EMB and (epoch + 1) % config.FREQ_EMB == 0:
            if len(tests) != 0:
                xs = np.r_[x_train, x_val, x_test]
                ys = np.r_[
                    y_train,
                    y_val,
                    np.full(shape=len(x_test), fill_value="test"),
                ]
            else:
                xs = np.r_[x_train, x_val]
                ys = np.r_[y_train, y_val]
            vis.latent_embed_plot_tsne(xs, ys)
            vis.latent_embed_plot_umap(xs, ys)

            if collect_meta:
                # merge img and rec into one image for display in altair
                meta_df["image"] = meta_df["image"].apply(vis.merge)
                vis.dyn_latentembed_plot(meta_df, epoch, embedding="umap")
                vis.dyn_latentembed_plot(meta_df, epoch, embedding="tsne")

        # visualise latent disentanglement
        if config.VIS_DIS and (epoch + 1) % config.FREQ_DIS == 0:
            if not pose:
                p_train = None
            vis.latent_disentamglement_plot(
                x_train, vae, device, poses=p_train
            )

        # visualise pose disentanglement
        if pose and config.VIS_POS and (epoch + 1) % config.FREQ_POS == 0:
            vis.pose_disentanglement_plot(x_train, p_train, vae, device)

        # visualise interpolations
        if config.VIS_INT and (epoch + 1) % config.FREQ_INT == 0:
            if len(tests) != 0:
                xs = np.r_[x_train, x_val, x_test]
                ys = np.r_[y_train, y_val, np.ones(len(x_test))]
                if pose:
                    ps = np.r_[p_train, p_val, p_test]
                else:
                    ps = None
            else:
                xs = np.r_[x_train, x_val]
                ys = np.r_[y_train, y_val]
                if pose:
                    ps = np.r_[p_train, p_val]
                else:
                    ps = None

            vis.interpolations_plot(
                xs, ys, vae, device, poses=ps  # do we need val and test here?
            )


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
