import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch

from . import settings, vis
from .data import load_data
from .utils import accuracy, latest_file
from .utils_learning import add_meta


def evaluate(
    datapath: str,
    datatype: str,
    state: str | None,
    meta: str | None,
    lim: int | None,
    splt: int,
    batch_s: int,
    classes: str | None,
    use_gpu: bool,
    gaussian_blur: bool,
    normalise: bool,
    shift_min: bool,
    rescale: bool,
    classifier: str,
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
    fabric = lt.Fabric()
    fabric.launch()
    # ############################### DATA ###############################
    tests, data_dim = load_data(
        datapath=datapath,
        datatype=datatype,
        lim=lim,
        splt=splt,
        batch_s=batch_s,
        eval=True,
        gaussian_blur=gaussian_blur,
        normalise=normalise,
        shift_min=shift_min,
        rescale=rescale,
        fabric=fabric,
    )

    # ############################### MODEL ###############################
    device = fabric.device

    if state is None:
        if not os.path.exists("states"):
            raise RuntimeError(
                "There are no existing model states saved or provided via the state flag in config unable to evaluate."
            )
        else:
            state = latest_file("states", ".pt")
            state = os.path.join("states", state)

    s = os.path.basename(state)
    fname = s.split(".")[0].split("_")
    dshape = list(tests)[0][0].shape[2:]
    pose_dims = int(fname[-1])

    logging.info("Loading model from: {}".format(state))
    checkpoint = torch.load(state)
    vae = checkpoint["model_class_object"]
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae = fabric.setup(vae)

    # ########################## EVALUATE ################################

    if meta is None:
        metas = latest_file("states", ".pkl")
        meta = os.path.join("states", metas)

    logging.info("Loading model from: {}".format(meta))
    meta_df = pd.read_pickle(meta)

    # create holders for latent spaces and labels
    x_test, y_test, c_test = [], [], []
    p_test = None

    if pose_dims != 0:
        p_test = []

    logging.debug("Batch: [0/%d]" % (len(tests)))

    vae.eval()
    for batch_number, (t, label, aff, meta_data) in enumerate(tests):

        # get data in the right device
        t, aff = t.to(device), aff.to(device)
        t = t.to(torch.float32)

        # forward
        t_hat, t_before_conv, t_mu, t_logvar, tlat, tlat_pose = vae(t)

        x_test.extend(t_mu.cpu().detach().numpy())  # store latents
        c_test.extend(t_logvar.cpu().detach().numpy())
        # if labels are present save them otherwise save test
        try:
            y_test.extend(label)
        except IndexError:
            np.full(shape=len(t), fill_value="test")
        if tlat_pose is not None:
            p_test.extend(tlat_pose.cpu().detach().numpy())

        meta_df = add_meta(
            data_dim,
            meta_df,
            meta_data,
            t_hat,
            t_mu,
            tlat_pose,
            tlat,
            mode="evl",
        )

        logging.debug("Batch: [%d/%d]" % (batch_number + 1, len(tests)))
    logging.info("Batch: [%d/%d]" % (batch_number + 1, len(tests)))

    # ########################## VISUALISE ################################
    if classes is not None:
        classes_list = pd.read_csv(classes).columns.tolist()
    else:
        classes_list = []
    # visualise reconstructions - last batch
    if settings.VIS_REC:
        vis.recon_plot(t, t_hat, y_test, data_dim, mode="evl")

    # visualise latent disentanglement
    if settings.VIS_DIS:
        vis.latent_disentamglement_plot(
            dshape,
            x_test,
            vae,
            device,
            poses=p_test,
            mode="_eval",
        )

    # visualise pose disentanglement
    if pose_dims != 0 and settings.VIS_POS:
        vis.pose_disentanglement_plot(
            dshape,
            x_test,
            p_test,
            vae,
            device,
            mode="_eval",
        )

    if pose_dims != 0 and settings.VIS_POSE_CLASS:
        vis.pose_class_disentanglement_plot(
            dshape,
            x_test,
            y_test,
            settings.VIS_POSE_CLASS,
            p_test,
            vae,
            device,
            mode="_eval",
        )
    # visualise interpolations
    if settings.VIS_INT:
        vis.interpolations_plot(
            dshape,
            x_test,
            np.ones(len(x_test)),
            vae,
            device,
            poses=p_test,
            mode="_eval",
        )

    # visualise embeddings
    if settings.VIS_EMB:
        vis.latent_embed_plot_umap(
            x_test, np.array(y_test), classes_list, "_eval"
        )
        vis.latent_embed_plot_tsne(
            x_test, np.array(y_test), classes_list, "_eval"
        )

    if settings.VIS_SIM:
        vis.latent_space_similarity_plot(
            x_test, np.array(y_test), mode="_eval", classes_order=classes_list
        )

    # ############################# Predict #############################
    # get training latent space from metadata for comparison and accuracy estimation
    latents_training = meta_df[meta_df["mode"] == "trn"][
        [col for col in meta_df if col.startswith("lat")]
    ].to_numpy()
    latents_training_id = meta_df[meta_df["mode"] == "trn"]["id"]

    if settings.VIS_DYN:
        # merge img and rec into one image for display in altair
        meta_df["image"] = meta_df["image"].apply(vis.merge)
        vis.dyn_latentembed_plot(meta_df, 0, embedding="umap", mode="_eval")
        vis.dyn_latentembed_plot(meta_df, 0, embedding="tsne", mode="_eval")

    # visualise embeddings
    if settings.VIS_EMB:
        vis.latent_embed_plot_umap(
            np.concatenate([x_test, latents_training]),
            np.concatenate([np.array(y_test), np.array(latents_training_id)]),
            classes_list,
            "_train_eval_comparison",
        )
        vis.latent_embed_plot_tsne(
            np.concatenate([x_test, latents_training]),
            np.concatenate([np.array(y_test), np.array(latents_training_id)]),
            classes_list,
            "_train_eval_comparison",
        )

    # visualise accuracy
    (train_acc, val_acc, val_acc_selected, ypred_train, ypred_val,) = accuracy(
        latents_training,
        np.array(latents_training_id),
        x_test,
        np.array(y_test),
        classifier=classifier,
    )
    logging.info(
        "------------------->>> Accuracy: Train: %f | Val : %f | Val with unseen labels: %f\n"
        % (train_acc, val_acc_selected, val_acc)
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
    logging.info("Saving meta files with evaluation data.")

    metas = os.path.basename(meta)
    # save metadata with evaluation data
    meta_df.to_pickle(
        os.path.join("states", metas.split(".")[0] + "_eval.pkl")
    )
