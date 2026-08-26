import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from . import settings, vis
from .cyc_annealing import setup_annealing
from .data import load_data
from .loss import AVAELoss
from .models import model_setup
from .utils import accuracy, as_list, latest_file
from .utils_gpu import setup_gpus
from .utils_learning import (
    build_meta_df,
    combine_meta_df,
    configure_optimiser,
    log_progress,
)


def train(
    datapath: str,
    datatype: str,
    restart: bool,
    state: str | None,
    lim: int | None,
    splt: int,
    batch_s: int,
    no_val_drop: bool,
    affinity: str | None,
    classes: str | None,
    epochs: int,
    channels: int,
    depth: int,
    filters: list | None,
    lat_dims: int,
    pose_dims: int,
    bnorm_encoder: bool,
    bnorm_decoder: bool,
    gsd_conv_layers: int,
    n_splats: int,
    klred: str,
    learning: float,
    beta_load: str | None,
    beta_min: float,
    beta_max: float,
    beta_cycle: int,
    beta_ratio: float,
    cyc_method_beta: str,
    gamma_load: str | None,
    gamma_min: float,
    gamma_max: float,
    gamma_cycle: int,
    gamma_ratio: float,
    cyc_method_gamma: str,
    recon_fn: str,
    use_gpu: bool,
    gpu_devices: str | None,
    model: str,
    opt_method: str,
    gaussian_blur: bool,
    normalise: bool,
    shift_min: bool,
    rescale: int | None,
    tensorboard: bool,
    classifier: str,
    strategy: str,
):
    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    datapath: str
        Path to the data directory.
    datatype: str
        data file formats : mrc, npy
    restart: bool
        If True, the model will be restarted from the latest saved state.
    state: str
        Path to the model state file to be used for evaluation/restart.
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
    epochs: int
        Number of epochs to train the model.
    channels: int
        Number of channels in the input data.
    depth: int
        Depth of the model.
    filters: list
        List of filters to use in the model.
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
    opt_method: str
        The method of optimisation. It can be adam/sgd/asgd (we can add other methods easily as we need them)
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise:
        In True, the input data is normalised before being passed to the model.
    shift_min: bool
        If True, the input data is shifted to have a minimum value of 0 and max of 1.
    tensorboard: bool
        If True, log metrics and figures using tensorboard.
    classifier: str
        The method to use on the latent space classification. Can be neural network (NN), k nearest neighbourgs (KNN) or logistic regression (LR).
    bnorm_encoder: bool
        If True, batch normalisation is applied to the encoder.
    bnorm_decoder: bool
        If True, batch normalisation is applied to the decoder.
    strategy: str
        The strategy to use for distributed training. Can be  'ddp', 'deepspeed' or 'fsdp".
    gsd_conv_layers: int
        activates convolution layers at the end of the differetiable decoder if set
        and it is an integer defining the number of output channels.
    n_splats: int
        The number of splats in the Gaussian Splat Decoder.
    klred: str
        The method to reduce the KL divergence. Can be 'mean' or 'sum'.
    beta_load: str
        Path to the beta values to load.
    gamma_load: str
        Path to the gamma values to load.
    rescale: int | None
        If provided, the data is rescaled by the value.
    """
    lt.pytorch.seed_everything(42)

    # ############################### LOGGING #################################

    writer = SummaryWriter() if tensorboard else None
    t_history = []
    v_history = []

    # ############################### GPU SETUP ################################

    fabric = setup_gpus(
        use_gpu=use_gpu,
        gpu_devices=gpu_devices,
        strategy=strategy,
    )

    fabric.launch()
    device = fabric.device
    rank_zero = fabric.global_rank == 0

    # ############################### ANNEALING ###############################

    beta_arr = setup_annealing(
        epochs=epochs,
        value_max=beta_max,
        value_min=beta_min,
        cyc_method=cyc_method_beta,
        n_cycle=beta_cycle,
        ratio=beta_ratio,
        cycle_load=beta_load,
    )

    gamma_arr = setup_annealing(
        epochs=epochs,
        value_max=gamma_max,
        value_min=gamma_min,
        cyc_method=cyc_method_gamma,
        n_cycle=gamma_cycle,
        ratio=gamma_ratio,
        cycle_load=gamma_load,
    )
    if settings.VIS_CYC:
        vis.plot_cyc_variable(beta_arr, "beta")
        vis.plot_cyc_variable(gamma_arr, "gamma")

    # ############################### DATA ###############################
    trains, vals, tests, affinity_matrix, data_dim = load_data(
        datapath=datapath,
        datatype=datatype,
        lim=lim,
        splt=splt,
        batch_s=batch_s,
        no_val_drop=no_val_drop,
        eval=False,
        affinity_path=affinity,
        classes=classes,
        gaussian_blur=gaussian_blur,
        normalise=normalise,
        shift_min=shift_min,
        rescale=rescale,
        fabric=fabric,
    )

    # The spacial dimensions of the data
    dshape = list(trains)[0][0].shape[2:]
    pose = not (pose_dims == 0)

    # ############################### MODEL ###############################
    vae = model_setup(
        model_type=model,
        input_shape=dshape,
        channels=channels,
        depth=depth,
        lat_dims=lat_dims,
        pose_dims=pose_dims,
        bnorm_encoder=bnorm_encoder,
        bnorm_decoder=bnorm_decoder,
        n_splats=n_splats,
        gsd_conv_layers=gsd_conv_layers,
        device=device,
        filters=filters,
    )

    logging.info(vae)

    # ################################# LOSS #################################

    loss = AVAELoss(
        device=device,
        beta=beta_arr,
        gamma=gamma_arr,
        lookup_aff=affinity_matrix,
        recon_fn=recon_fn,
        klred=klred,
    )

    # ############################### OPTIMISER ###############################
    optimizer = configure_optimiser(
        opt_method=opt_method, model=vae, learning_rate=learning
    )

    vae, optimizer = fabric.setup(vae, optimizer)

    # ############################### RESTARTS ################################
    e_start = 0

    if restart:
        if state is None:
            if not os.path.exists("states"):
                raise RuntimeError(
                    "There are no existing model states saved or provided via the state flag in config unable to evaluate."
                )
            else:
                state = latest_file("states", ".pt")
                state = os.path.join("states", state)

        checkpoint = torch.load(state, weights_only=False)
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        e_start = checkpoint["epoch"]
        t_history = checkpoint["t_loss_history"]
        v_history = checkpoint["v_loss_history"]

    # ########################## TRAINING LOOP ################################
    for epoch in range(e_start, epochs):

        # populate loss with new epoch
        t_history.append(np.zeros(4))
        v_history.append(np.zeros(4))

        # Per-rank epoch metadata buffers grouped by mode.
        filename_train, filename_val, filename_test = [], [], []
        meta_train, meta_val, meta_test = [], [], []

        # create holders for data, labels, and latent spaces
        x_train, x_val, x_test = [], [], []
        xhat_train, xhat_val, xhat_test = [], [], []

        y_train, y_val, y_test = [], [], []
        z_train, z_val, z_test = [], [], []
        c_train, c_val, c_test = [], [], []

        if pose:
            p_train, p_val, p_test = [], [], []

        # ########################## TRAINING #################################
        vae.train()
        for batch_number, (x, ys, aff, meta_data) in enumerate(trains):

            # get data in the right device
            x, aff = x.to(device), aff.to(device)
            x = x.to(torch.float32)

            # forward
            x_hat, lat_mu, lat_logvar, lat, lat_pose = vae(x)
            history_loss = loss(
                x, x_hat, lat_mu, lat_logvar, epoch, batch_aff=aff
            )

            # record loss
            for i in range(len(t_history[-1])):
                t_history[-1][i] += history_loss[i].item()
            log_progress(
                "Epoch: [%d/%d] | Batch: [%d/%d]"
                % (epoch + 1, epochs, batch_number + 1, len(trains))
            )

            # backwards
            fabric.backward(history_loss[0])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            z_train.extend(lat_mu.cpu().detach().numpy())  # store latents
            y_train.extend(as_list(ys))
            c_train.extend(lat_logvar.cpu().detach().numpy())
            if pose:
                p_train.extend(lat_pose.cpu().detach().numpy())

            filename_train.extend(as_list(meta_data.get("filename", [])))
            meta_train.extend(as_list(meta_data.get("meta", [])))
            x_train.extend(as_list(meta_data.get("image", [])))
            xhat_train.extend(vis.format(x_hat, data_dim))

        t_history[-1] /= len(trains)

        logging.info(
            "Training : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
            "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f"
            % (
                epoch + 1,
                epochs,
                *t_history[-1],
                beta_arr[epoch],
                gamma_arr[epoch],
            )
        )
        # ########################## VAL ######################################
        vae.eval()
        with torch.inference_mode():
            for batch_number, (v, ys, aff, meta_data) in enumerate(vals):

                # get data in the right device
                v, aff = v.to(device), aff.to(device)
                v = v.to(torch.float32)

                # forward
                v_hat, v_mu, v_logvar, vlat, vlat_pos = vae(v)
                v_history_loss = loss(
                    v, v_hat, v_mu, v_logvar, epoch, batch_aff=aff
                )

                # record loss
                for i in range(len(t_history[-1])):
                    v_history[-1][i] += v_history_loss[i].item()
                log_progress(
                    "Epoch: [%d/%d] | Batch: [%d/%d]"
                    % (epoch + 1, epochs, batch_number + 1, len(vals))
                )

                z_val.extend(v_mu.cpu().detach().numpy())  # store latents
                y_val.extend(as_list(ys))
                c_val.extend(v_logvar.cpu().detach().numpy())
                if pose:
                    p_val.extend(vlat_pos.cpu().detach().numpy())

                filename_val.extend(as_list(meta_data.get("filename", [])))
                meta_val.extend(as_list(meta_data.get("meta", [])))
                x_val.extend(as_list(meta_data.get("image", [])))
                xhat_val.extend(vis.format(v_hat, data_dim))
            v_history[-1] /= len(vals)

            logging.info(
                "Validation : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
                "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f"
                % (
                    epoch + 1,
                    epochs,
                    *v_history[-1],
                    beta_arr[epoch],
                    gamma_arr[epoch],
                )
            )

            if writer:
                for i, loss_name in enumerate(
                    ["Loss", "Recon loss", "KLdiv loss", "Affin loss"]
                ):
                    writer.add_scalar(loss_name, v_history[-1][i], epoch)

            # ########################## TEST #####################################
            if (epoch + 1) % settings.FREQ_EVAL == 0:
                for batch_number, (t, ys, aff, meta_data) in enumerate(
                    tests
                ):  # tests empty if no 'test' dir
                    # get data in the right device
                    t, aff = t.to(device), aff.to(device)
                    t = t.to(torch.float32)

                    # forward
                    t_hat, t_mu, t_logvar, tlat, tlat_pose = vae(t)

                    z_test.extend(t_mu.cpu().detach().numpy())  # store latents
                    y_test.extend(as_list(ys))
                    c_test.extend(t_logvar.cpu().detach().numpy())
                    if pose:
                        p_test.extend(tlat_pose.cpu().detach().numpy())

                    filename_test.extend(
                        as_list(meta_data.get("filename", []))
                    )
                    meta_test.extend(as_list(meta_data.get("meta", [])))
                    x_test.extend(as_list(meta_data.get("image", [])))
                    xhat_test.extend(vis.format(t_hat, data_dim))

                    log_progress(
                        "Epoch: [%d/%d] | Batch: [%d/%d]"
                        % (epoch + 1, epochs, batch_number + 1, len(tests))
                    )
                logging.info(
                    "Evaluation : Epoch: [%d/%d]" % (epoch + 1, epochs)
                )
            logging.info("\n")  # end of training round

        needs_meta_df = (
            rank_zero
            and settings.VIS_EMB
            and settings.VIS_DYN
            and (epoch + 1) % settings.FREQ_EMB == 0
        ) or ((epoch + 1) % settings.FREQ_STA == 0)
        if needs_meta_df:
            meta_df = build_meta_df(
                pose=pose,
                train={
                    "filename": filename_train,
                    "meta": meta_train,
                    "x": x_train,
                    "xhat": xhat_train,
                    "y": y_train,
                    "z": z_train,
                    "logvar": c_train,
                    "pose": p_train if pose else None,
                },
                val={
                    "filename": filename_val,
                    "meta": meta_val,
                    "x": x_val,
                    "xhat": xhat_val,
                    "y": y_val,
                    "z": z_val,
                    "logvar": c_val,
                    "pose": p_val if pose else None,
                },
                test={
                    "filename": filename_test,
                    "meta": meta_test,
                    "x": x_test,
                    "xhat": xhat_test,
                    "y": y_test,
                    "z": z_test,
                    "logvar": c_test,
                    "pose": p_test if pose else None,
                },
            )

            # Dynamic plots and save should use metadata from all ranks.
            combined_meta_df = combine_meta_df(
                meta_df=meta_df,
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )

        # ########################## VISUALISE ################################

        if classes is not None:
            classes_list = pd.read_csv(classes).columns.tolist()
        else:
            classes_list = []

        # visualise accuracy: confusion and F1 scores
        if (
            rank_zero
            and settings.VIS_ACC
            and (epoch + 1) % settings.FREQ_ACC == 0
        ):
            train_acc, val_acc, _, ypred_train, ypred_val = accuracy(
                z_train, y_train, z_val, y_val, classifier=classifier
            )

            logging.info(
                "------------------->>> Accuracy: Train: %f | Val: %f\n"
                % (train_acc, val_acc),
            )
            vis.accuracy_plot(
                y_train,
                ypred_train,
                y_val,
                ypred_val,
                classes,
                epoch=epoch,
                writer=writer,
            )

            vis.f1_plot(
                y_train,
                ypred_train,
                y_val,
                ypred_val,
                classes,
                epoch=epoch,
                writer=writer,
            )

        # visualise loss
        if rank_zero and settings.VIS_LOS and epoch > 0:
            p = [
                len(trains),
                depth,
                channels,
                lat_dims,
                learning,
                beta_arr[epoch],
                gamma_arr[epoch],
            ]
            vis.loss_plot(
                epoch + 1,
                beta_arr[: epoch + 1],
                gamma_arr[: epoch + 1],
                t_history,
                v_history,
                p=p,
            )

        # visualise reconstructions - last batch
        if (
            rank_zero
            and settings.VIS_REC
            and (epoch + 1) % settings.FREQ_REC == 0
        ):
            vis.recon_plot(
                x,
                x_hat,
                y_train,
                data_dim,
                mode="trn",
                epoch=epoch,
                writer=writer,
            )
            vis.recon_plot(
                v,
                v_hat,
                y_val,
                data_dim,
                mode="val",
                epoch=epoch,
                writer=writer,
            )

        # visualise mean and logvar similarity matrix
        if (
            rank_zero
            and settings.VIS_SIM
            and (epoch + 1) % settings.FREQ_SIM == 0
        ):
            vis.latent_space_similarity_plot(
                z_train,
                np.array(y_train),
                mode="_train",
                epoch=epoch,
                classes_order=classes_list,
            )
            vis.latent_space_similarity_plot(
                z_val,
                np.array(y_val),
                mode="_valid",
                epoch=epoch,
                classes_order=classes_list,
            )

        # visualise embeddings
        if (
            rank_zero
            and settings.VIS_EMB
            and (epoch + 1) % settings.FREQ_EMB == 0
        ):
            if len(tests) != 0:
                xs = np.r_[z_train, z_val, z_test]
                ys = np.r_[
                    y_train,
                    y_val,
                    np.full(shape=len(z_test), fill_value="test"),
                ]
                if pose:
                    ps = np.r_[p_train, p_val, p_test]
            else:
                xs = np.r_[z_train, z_val]
                ys = np.r_[y_train, y_val]
                if pose:
                    ps = np.r_[p_train, p_val]

            vis.latent_embed_plot_tsne(xs, ys, epoch=epoch, writer=writer)
            vis.latent_embed_plot_umap(
                xs, ys, classes_list, epoch=epoch, writer=writer
            )
            if pose:
                vis.latent_embed_plot_tsne(
                    ps, ys, epoch=epoch, writer=writer, mode="pose"
                )
                vis.latent_embed_plot_umap(
                    ps, ys, epoch=epoch, writer=writer, mode="pose"
                )

            if settings.VIS_DYN:
                # merge img and rec into one image for display in altair
                combined_meta_df["image"] = combined_meta_df["image"].apply(
                    vis.merge
                )
                vis.dyn_latentembed_plot(
                    combined_meta_df, epoch, embedding="umap"
                )
                vis.dyn_latentembed_plot(
                    combined_meta_df, epoch, embedding="tsne"
                )

        # visualise latent disentanglement
        if (
            rank_zero
            and settings.VIS_DIS
            and (epoch + 1) % settings.FREQ_DIS == 0
        ):
            if not pose:
                poses = None
            else:
                poses = p_train
            vis.latent_disentamglement_plot(
                dshape,
                z_train,
                vae,
                device,
                poses=poses,
            )

        # visualise pose disentanglement
        if (
            rank_zero
            and pose
            and settings.VIS_POS
            and (epoch + 1) % settings.FREQ_POS == 0
        ):
            vis.pose_disentanglement_plot(
                dshape,
                z_train,
                p_train,
                vae,
                device,
            )

            if settings.VIS_POSE_CLASS is not None:
                vis.pose_class_disentanglement_plot(
                    dshape,
                    z_train,
                    y_train,
                    settings.VIS_POSE_CLASS,
                    p_train,
                    vae,
                    device,
                )

        # visualise interpolations
        if (
            rank_zero
            and settings.VIS_INT
            and (epoch + 1) % settings.FREQ_INT == 0
        ):
            if len(tests) != 0:
                xs = np.r_[z_train, z_val, z_test]
                ys = np.r_[y_train, y_val, np.ones(len(z_test))]
                if pose:
                    ps = np.r_[p_train, p_val, p_test]
                else:
                    ps = None
            else:
                xs = np.r_[z_train, z_val]
                ys = np.r_[y_train, y_val]
                if pose:
                    ps = np.r_[p_train, p_val]
                else:
                    ps = None

            if settings.VIS_Z_N_INT is not None:
                vis.latent_4enc_interpolate_plot(
                    dshape,
                    xs,
                    ys,
                    vae,
                    device,
                    settings.VIS_Z_N_INT,
                    poses=ps,
                )

            vis.interpolations_plot(
                dshape,
                xs,
                ys,
                vae,
                device,
                poses=ps,  # do we need val and test here?
            )
        # ########################## SAVE STATE ###############################
        if (epoch + 1) % settings.FREQ_STA == 0:
            if rank_zero:
                if not os.path.exists("states"):
                    os.mkdir("states")

                mname = (
                    "avae_"
                    + str(settings.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_"
                    + str(lat_dims)
                    + "_"
                    + str(pose_dims)
                    + ".pt"
                )

                logging.info(
                    "################################################################"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": vae._original_module.state_dict(),
                        "model_class_object": vae._original_module,
                        "optimizer_state_dict": optimizer._optimizer.state_dict(),
                        "t_loss_history": t_history,
                        "v_loss_history": v_history,
                    },
                    os.path.join("states", mname),
                )
                logging.info(
                    f"Saved model state: {mname} for restarting and evaluation "
                )

                filename = (
                    "meta_"
                    + str(settings.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_"
                    + str(lat_dims)
                    + "_"
                    + str(pose_dims)
                    + ".pkl"
                )
                combined_meta_df.to_pickle(os.path.join("states", filename))

                logging.info(f"Saved meta file : {filename} for evaluation \n")
        fabric.barrier()

    if writer:
        writer.flush()
        writer.close()
