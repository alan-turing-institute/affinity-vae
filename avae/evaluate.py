import logging
import os

import numpy as np
import pandas as pd
import torch

from . import config, vis
from .data import load_data
from .utils import accuracy
from .utils_learning import add_meta, pass_batch, set_device


def evaluate(
    datapath,
    datatype,
    state,
    meta,
    lim,
    splt,
    batch_s,
    classes,
    collect_meta,
    use_gpu,
    gaussian_blur,
    normalise,
    shift_min,
    classifier,
):
    """Function for evaluating the model. Loads the data, model and runs the evaluation. Saves the results of the
    evaluation in the plot and latents directories.

    Parameters
    ----------
    datapath: str
        Path to the data directory.
    datatype: str
        data file formats : mrc, npy
    state: str
        Path to the model state file to be used for evaluation/resume.
    meta: str
        Path to the meta file to be used for evaluation/resume.
    lim: int
        Limit the number of samples to load.
    splt: int
        Percentage of data to be used for validation.
    batch_s: int
        Batch size.
    classes: list
        List of classes to be selected from the data for the training and validation set.
    collect_meta: bool
        If True, the meta data for visualisation will be collected and returned.
    use_gpu: bool
        If True, the model will be trained on GPU.
    gaussian_blur: bool
        if True, Gaussian bluring is applied to the input before being passed to the model.
        This is added as a way to remove noise from the input data.
    normalise:
        In True, the input data is normalised before being passed to the model.
    shift_min: bool
        If True, the input data is shifted to have a minimum value of 0 and max 1.
    classifier: str
        The method to use on the latent space classification. Can be neural network (NN), k nearest neighbourgs (KNN) or logistic regression (LR).


    """
    # ############################### DATA ###############################
    tests, data_dim = load_data(
        datapath,
        datatype,
        lim,
        splt,
        batch_s,
        collect_meta=collect_meta,
        eval=True,
        gaussian_blur=gaussian_blur,
        normalise=normalise,
        shift_min=shift_min,
    )

    # ############################### MODEL ###############################
    device = set_device(use_gpu)

    if state is None:
        if not os.path.exists("states"):
            raise RuntimeError(
                "There are no existing model states saved or provided via the state flag in config unable to evaluate."
            )
        else:
            state = sorted(
                [s for s in os.listdir("states") if ".pt" in s],
                key=lambda x: int(x.split("_")[2][1:]),
            )[-1]
            state = os.path.join("states", state)

    fname = state.split(".")[0].split("_")
    pose_dims = fname[3]

    logging.info("Loading model from: ", state, flush=True)
    checkpoint = torch.load(state)
    vae = checkpoint["model_class_object"]
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.to(device)

    # ########################## EVALUATE ################################

    if meta is None:
        if collect_meta:
            metas = sorted(
                [f for f in os.listdir("states") if ".pkl" in f],
                key=lambda x: int(x.split("_")[2][1:]),
            )[-1]
            meta = os.path.join("states", metas)

    meta_df = pd.read_pickle(meta)

    # create holders for latent spaces and labels
    x_test = []
    y_test = []
    c_test = []

    if pose_dims != 0:
        p_test = []

    logging.info("Batch: [0/%d]" % (len(tests)), end="\r", flush=True)

    vae.eval()
    for b, batch in enumerate(tests):
        x, x_hat, lat_mu, lat_logvar, lat, lat_pose, _ = pass_batch(
            device, vae, batch, b, len(tests)
        )
        x_test.extend(lat_mu.cpu().detach().numpy())
        c_test.extend(lat_logvar.cpu().detach().numpy())

        # if labels are present save them otherwise save test
        try:
            y_test.extend(batch[1])
        except IndexError:
            np.full(shape=len(batch[0]), fill_value="test")
        if pose_dims != 0:
            p_test.extend(lat_pose.cpu().detach().numpy())

        if collect_meta:  # store meta for plots
            meta_df = add_meta(
                data_dim,
                meta_df,
                batch[-1],
                x_hat,
                lat_mu,
                lat_pose,
                lat_logvar,
                mode="evl",
            )

        logging.info(
            "Batch: [%d/%d]" % (b + 1, len(tests)), end="\r", flush=True
        )
    logging.info("Batch: [%d/%d]" % (b + 1, len(tests)), flush=True)

    # ########################## VISUALISE ################################

    # visualise reconstructions - last batch
    if config.VIS_REC:
        vis.recon_plot(x, x_hat, y_test, data_dim, mode="evl")

    # visualise latent disentanglement
    if config.VIS_DIS:
        vis.latent_disentamglement_plot(
            x_test, vae, device, data_dim, poses=p_test, mode="_eval"
        )

    # visualise pose disentanglement
    if pose_dims != 0 and config.VIS_POS:
        vis.pose_disentanglement_plot(
            x_test, p_test, vae, data_dim, device, mode="_eval"
        )

    # visualise interpolations
    if config.VIS_INT:
        vis.interpolations_plot(
            x_test,
            np.ones(len(x_test)),
            vae,
            device,
            data_dim,
            poses=p_test,
            mode="_eval",
        )

    # visualise embeddings
    if config.VIS_EMB:
        vis.latent_embed_plot_umap(x_test, np.array(y_test), "_eval")
        vis.latent_embed_plot_tsne(x_test, np.array(y_test), "_eval")

    if config.VIS_SIM:
        if classes is not None:
            classes_list = pd.read_csv(classes).columns.tolist()
        else:
            classes_list = []
        vis.latent_space_similarity(
            x_test, np.array(y_test), mode="_eval", classes_order=classes_list
        )

    # ############################# Predict #############################

    if collect_meta:
        # merge img and rec into one image for display in altair
        meta_df["image"] = meta_df["image"].apply(vis.merge)
        vis.dyn_latentembed_plot(meta_df, 0, embedding="umap", mode="_eval")
        vis.dyn_latentembed_plot(meta_df, 0, embedding="tsne", mode="_eval")

        # get training latent space from metadata for comparison and accuracy estimation
        latents_training = meta_df[meta_df["mode"] == "trn"][
            [col for col in meta_df if col.startswith("lat")]
        ].to_numpy()
        latents_training_id = meta_df[meta_df["mode"] == "trn"]["id"]

        # visualise embeddings
        if config.VIS_EMB:
            vis.latent_embed_plot_umap(
                np.concatenate([x_test, latents_training]),
                np.concatenate(
                    [np.array(y_test), np.array(latents_training_id)]
                ),
                "_train_eval_comparison",
            )
            vis.latent_embed_plot_tsne(
                np.concatenate([x_test, latents_training]),
                np.concatenate(
                    [np.array(y_test), np.array(latents_training_id)]
                ),
                "_train_eval_comparison",
            )

        # visualise accuracy
        train_acc, val_acc, ypred_train, ypred_val = accuracy(
            latents_training,
            np.array(latents_training_id),
            x_test,
            np.array(y_test),
            classifier=classifier,
        )
        logging.info(
            "\n------------------->>> Accuracy: Train: %f | Val: %f\n"
            % (train_acc, val_acc)
        )
        vis.accuracy_plot(
            np.array(latents_training_id),
            ypred_train,
            y_test,
            ypred_val,
            classes,
            mode="_eval",
        )
        vis.f1_plot(
            np.array(latents_training_id),
            ypred_train,
            y_test,
            ypred_val,
            classes,
            mode="_eval",
        )

        # save metadata with evaluation data
        meta_df.to_pickle(
            os.path.join("states", metas.split(".")[0] + "_eval.pkl")
        )
