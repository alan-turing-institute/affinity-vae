import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from avae.decoders.decoders import Decoder, DecoderA, DecoderB
from avae.decoders.differentiable import GaussianSplatDecoder
from avae.encoders.encoders import Encoder, EncoderA, EncoderB

from . import settings, vis
from .cyc_annealing import cyc_annealing
from .data import load_data
from .loss import AVAELoss
from .models import AffinityVAE
from .utils import accuracy, latest_file
from .utils_learning import add_meta, pass_batch, set_device


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
    model: str,
    opt_method: str,
    gaussian_blur: bool,
    normalise: bool,
    shift_min: bool,
    rescale: bool,
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
    bnrom_decoder: bool
        If True, batch normalisation is applied to the decoder.
    strategy: str
        The strategy to use for distributed training. Can be  'ddp', 'deepspeed' or 'fsdp".
    gsd_conv_layers: int
        activates convolution layers at the end of the differetiable decoder if set
        and it is an integer defining the number of output channels  .
    """
    lt.pytorch.seed_everything(42)

    n_devices = torch.cuda.device_count()
    logging.info('GPus available: {}'.format(n_devices))

    if n_devices > 0 and use_gpu is True:
        accelerator = 'gpu'

        if n_devices <= 4:
            n_nodes = 1
        else:
            # Calculate the number of nodes based on the formula: ceil(num_gpus / 4)
            n_nodes = (n_devices + 3) // 4

        logging.info(
            f'Setting up fabric with strategy {strategy}, accelerator {accelerator}, devices {n_devices}, num_nodes {n_nodes}'
        )
        fabric = lt.Fabric(
            strategy=strategy,
            accelerator=accelerator,
            devices=n_devices,
            num_nodes=n_nodes,
            # plugins=[SLURMEnvironment(auto_requeue=False)]
        )

    else:
        fabric = lt.Fabric(strategy=strategy, accelerator='auto')

    fabric.launch()
    device = fabric.device

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
    if filters is not None:
        filters = np.array(
            np.array(filters).replace(" ", "").split(","), dtype=np.int64
        )

    if model == "a":
        encoder = EncoderA(
            dshape, channels, depth, lat_dims, pose_dims, bnorm=bnorm_encoder
        )
        decoder = DecoderA(
            dshape, channels, depth, lat_dims, pose_dims, bnorm=bnorm_decoder
        )
    elif model == "b":
        encoder = EncoderB(dshape, channels, depth, lat_dims, pose_dims)
        decoder = DecoderB(dshape, channels, depth, lat_dims, pose_dims)
    elif model == "u":
        encoder = Encoder(
            input_size=dshape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=lat_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_encoder,
        )
        decoder = Decoder(
            input_size=dshape,
            capacity=channels,
            filters=filters,
            depth=depth,
            latent_dims=lat_dims,
            pose_dims=pose_dims,
            bnorm=bnorm_decoder,
        )
    elif model == "gsd":
        encoder = EncoderA(
            dshape, channels, depth, lat_dims, pose_dims, bnorm=bnorm_encoder
        )
        decoder = GaussianSplatDecoder(
            dshape,
            n_splats=n_splats,
            latent_dims=lat_dims,
            output_channels=gsd_conv_layers,
            device=device,
            pose_dims=pose_dims,
        )
    else:
        raise ValueError(
            "Invalid model type",
            model,
            "must be one of : a, b, u or gsd",
        )

    vae = AffinityVAE(encoder, decoder)

    logging.info(vae)

    if opt_method == "adam":
        optimizer = torch.optim.Adam(
            params=vae.parameters(), lr=learning  # , weight_decay=1e-5
        )
    elif opt_method == "sgd":
        optimizer = torch.optim.SGD(
            params=vae.parameters(), lr=learning  # , weight_decay=1e-5
        )
    elif opt_method == "asgd":
        optimizer = torch.optim.aSGD(
            params=vae.parameters(), lr=learning  # , weight_decay=1e-5
        )
    else:
        raise ValueError(
            "Invalid optimisation method",
            opt_method,
            "must be adam or sgd if you have other methods in mind, this can be easily added to the train.py",
        )

    vae, optimizer = fabric.setup(vae, optimizer)

    t_history = []
    v_history = []
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

        checkpoint = torch.load(state)
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        e_start = checkpoint["epoch"]
        t_history = checkpoint["t_loss_history"]
        v_history = checkpoint["v_loss_history"]

    if beta_max == 0 and cyc_method_beta != "flat" and beta_load is not None:
        raise RuntimeError(
            "The maximum value for beta is set to 0, it is not possible to"
            "oscillate between a maximum and minimum. Please choose the flat method for"
            "cyc_method_beta"
        )

    if beta_load is None:
        # If a path for loading the beta array is not provided,
        # create it given the input
        beta_arr = (
            cyc_annealing(
                epochs,
                cyc_method_beta,
                n_cycle=beta_cycle,
                ratio=beta_ratio,
            ).var
            * (beta_max - beta_min)
            + beta_min
        )
    else:
        beta_arr = np.load(beta_load)
        if len(beta_arr) != epochs:
            raise RuntimeError(
                f"The length of the beta array loaded from file is {len(beta_arr)} but the number of Epochs specified in the input are {epochs}.\n"
                "These two values should be the same."
            )

    if (
        gamma_max == 0
        and cyc_method_gamma != "flat"
        and gamma_load is not None
    ):
        raise RuntimeError(
            "The maximum value for gamma is set to 0, it is not possible to"
            "oscillate between a maximum and minimum. Please choose the flat method for"
            "cyc_method_gamma"
        )

    if gamma_load is None:
        # If a path for loading the gamma array is not provided,
        # create it given the input
        gamma_arr = (
            cyc_annealing(
                epochs,
                cyc_method_gamma,
                n_cycle=gamma_cycle,
                ratio=gamma_ratio,
            ).var
            * (gamma_max - gamma_min)
            + gamma_min
        )
    else:
        gamma_arr = np.load(gamma_load)
        if len(gamma_arr) != epochs:
            raise RuntimeError(
                f"The length of the gamma array loaded from file is {len(gamma_arr)} but the number of Epochs specified in the input are {epochs}.\n"
                "These two values should be the same."
            )
    if settings.VIS_CYC:
        vis.plot_cyc_variable(beta_arr, "beta")
        vis.plot_cyc_variable(gamma_arr, "gamma")

    loss = AVAELoss(
        device=device,
        beta=beta_arr,
        gamma=gamma_arr,
        lookup_aff=affinity_matrix,
        recon_fn=recon_fn,
        klred=klred,
    )

    if tensorboard:
        writer = SummaryWriter()
    else:
        writer = None

    # ########################## TRAINING LOOP ################################
    for epoch in range(e_start, epochs):

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

            (
                x,
                x_hat,
                lat_mu,
                lat_logvar,
                lat,
                lat_pos,
                t_history,
            ) = pass_batch(
                fabric,
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

            # store meta for plots and accuracy
            meta_df = add_meta(
                data_dim,
                meta_df,
                batch[-1],
                x_hat,
                lat_mu,
                lat_pos,
                lat_logvar,
                mode="trn",
            )

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
        for b, batch in enumerate(vals):
            (
                v,
                v_hat,
                v_mu,
                v_logvar,
                vlat,
                vlat_pos,
                v_history,
            ) = pass_batch(
                fabric,
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

            meta_df = add_meta(
                data_dim,
                meta_df,
                batch[-1],
                v_hat,
                v_mu,
                vlat_pos,
                v_logvar,
                mode="val",
            )
        v_history[-1] /= len(vals)

        logging.info(
            "Validation : Epoch: [%d/%d] |Loss: %f | Recon: %f | "
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
            for b, batch in enumerate(tests):  # tests empty if no 'test' dir
                (t, t_hat, t_mu, t_logvar, tlat, tlat_pose, _,) = pass_batch(
                    fabric, vae, batch, b, len(tests), epoch, epochs
                )
                x_test.extend(t_mu.cpu().detach().numpy())  # store latents
                c_test.extend(t_logvar.cpu().detach().numpy())
                if pose:
                    p_test.extend(tlat_pose.cpu().detach().numpy())

                # store meta for plots and classification
                meta_df = add_meta(
                    data_dim,
                    meta_df,
                    batch[-1],
                    t_hat,
                    t_mu,
                    tlat_pose,
                    t_logvar,
                    mode="tst",
                )

            logging.info("Evaluation : Batch: [%d/%d]" % (b + 1, len(tests)))
        logging.info("\n")  # end of training round

        # ########################## VISUALISE ################################

        if classes is not None:
            classes_list = pd.read_csv(classes).columns.tolist()
        else:
            classes_list = []

        # visualise accuracy: confusion and F1 scores
        if settings.VIS_ACC and (epoch + 1) % settings.FREQ_ACC == 0:
            train_acc, val_acc, _, ypred_train, ypred_val = accuracy(
                x_train, y_train, x_val, y_val, classifier=classifier
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
        if settings.VIS_LOS and epoch > 0:
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
        if settings.VIS_REC and (epoch + 1) % settings.FREQ_REC == 0:
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
        if settings.VIS_SIM and (epoch + 1) % settings.FREQ_SIM == 0:

            vis.latent_space_similarity_plot(
                x_train,
                np.array(y_train),
                mode="_train",
                epoch=epoch,
                classes_order=classes_list,
            )
            vis.latent_space_similarity_plot(
                x_val,
                np.array(y_val),
                mode="_valid",
                epoch=epoch,
                classes_order=classes_list,
            )

        # visualise embeddings
        if settings.VIS_EMB and (epoch + 1) % settings.FREQ_EMB == 0:
            if len(tests) != 0:
                xs = np.r_[x_train, x_val, x_test]
                ys = np.r_[
                    y_train,
                    y_val,
                    np.full(shape=len(x_test), fill_value="test"),
                ]
                if pose:
                    ps = np.r_[p_train, p_val, p_test]
            else:
                xs = np.r_[x_train, x_val]
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
                meta_df["image"] = meta_df["image"].apply(vis.merge)
                vis.dyn_latentembed_plot(meta_df, epoch, embedding="umap")
                vis.dyn_latentembed_plot(meta_df, epoch, embedding="tsne")

        # visualise latent disentanglement
        if settings.VIS_DIS and (epoch + 1) % settings.FREQ_DIS == 0:
            if not pose:
                poses = None
            else:
                poses = p_train
            vis.latent_disentamglement_plot(
                dshape,
                x_train,
                vae,
                device,
                poses=poses,
            )

        # visualise pose disentanglement
        if pose and settings.VIS_POS and (epoch + 1) % settings.FREQ_POS == 0:
            vis.pose_disentanglement_plot(
                dshape,
                x_train,
                p_train,
                vae,
                device,
            )

            if settings.VIS_POSE_CLASS is not None:
                vis.pose_class_disentanglement_plot(
                    dshape,
                    x_train,
                    y_train,
                    settings.VIS_POSE_CLASS,
                    p_train,
                    vae,
                    device,
                )

        # visualise interpolations
        if settings.VIS_INT and (epoch + 1) % settings.FREQ_INT == 0:
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
            meta_df.to_pickle(os.path.join("states", filename))

            logging.info(f"Saved meta file : {filename} for evaluation \n")

    if writer:
        writer.flush()
        writer.close()
