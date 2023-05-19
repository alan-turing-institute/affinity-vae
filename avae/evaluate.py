import os

import numpy as np
import pandas as pd
import torch

from . import config, vis
from .data import load_data
from .train import accuracy, add_meta, pass_batch
from .utils import set_device


def evaluate(datapath, lim, splt, batch_s, collect_meta, use_gpu):

    # ############################### DATA ###############################
    tests = load_data(
        datapath, lim, splt, batch_s, collect_meta=collect_meta, eval=True
    )

    # ############################### MODEL ###############################
    device = set_device(use_gpu)

    if not os.path.exists("states"):
        raise RuntimeError(
            "There are no existing model states saved, unable to evaluate."
        )
    # TODO add param to chose model
    states = sorted([s for s in os.listdir("states") if ".pt" in s])[0]
    fname = states.split(".")[0].split("_")
    pose_dims = fname[3]
    vae = torch.load(os.path.join("states", states))  # make optional param

    vae.to(device)

    # ########################## EVALUATE ################################
    metas = sorted([f for f in os.listdir("states") if ".pkl" in f])[-1]
    meta_df = pd.read_pickle(os.path.join("states", metas))

    # create holders for latent spaces and labels
    x_test = []
    y_test = []

    if pose_dims != 0:
        p_test = []

    print("Batch: [0/%d]" % (len(tests)), end="\r")

    vae.eval()
    for b, batch in enumerate(tests):
        x, x_hat, lat_mu, lat_logvar, lat, lat_pose, _ = pass_batch(
            device, vae, batch, b, len(tests)
        )
        x_test.extend(lat_mu.cpu().detach().numpy())

        # if labels are present save them otherwise save test
        try:
            y_test.extend(batch[1])
        except IndexError:
            np.full(shape=len(batch[0]), fill_value="test")
        if pose_dims != 0:
            p_test.extend(lat_pose.cpu().detach().numpy())

        if collect_meta:  # store meta for plots
            meta_df = add_meta(
                meta_df, batch[-1], x_hat, lat_mu, lat_pose, mode="evl"
            )

        print("Batch: [%d/%d]" % (b + 1, len(tests)), end="\r")
    print("Batch: [%d/%d]" % (b + 1, len(tests)))

    # ########################## VISUALISE ################################

    # visualise reconstructions - last batch
    if config.VIS_REC:
        vis.recon_plot(x, x_hat, name="evl")

    # get training latent space
    latents_training = meta_df[meta_df["mode"] == "trn"][
        [col for col in meta_df if col.startswith("lat")]
    ].to_numpy()
    latents_training_id = meta_df[meta_df["mode"] == "trn"]["id"]

    # visualise embeddings
    if config.VIS_EMB:
        vis.latent_embed_plot_umap(
            np.concatenate([x_test, latents_training]),
            np.concatenate([np.array(y_test), np.array(latents_training_id)]),
            "_eval",
        )
        vis.latent_embed_plot_tsne(
            np.concatenate([x_test, latents_training]),
            np.concatenate([np.array(y_test), np.array(latents_training_id)]),
            "_eval",
        )

        if collect_meta:
            # merge img and rec into one image for display in altair
            meta_df["image"] = meta_df["image"].apply(vis.merge)
            vis.dyn_latentembed_plot(meta_df, 0, embedding="umap")
            vis.dyn_latentembed_plot(meta_df, 0, embedding="tsne")

    # visualise accuracy
    train_acc, val_acc, ypred_train, ypred_val = accuracy(
        latents_training,
        np.array(latents_training_id),
        x_test,
        np.array(y_test),
    )
    print(
        "\n------------------->>> Accuracy: Train: %f | Val: %f\n"
        % (train_acc, val_acc)
    )
    vis.accuracy_plot(
        np.array(latents_training_id),
        ypred_train,
        y_test,
        ypred_val,
        title="_eval",
    )

    # visualise latent disentanglement
    if config.VIS_DIS:
        vis.latent_disentamglement_plot(x_test, vae, device, poses=p_test)

    # visualise pose disentanglement
    if pose_dims != 0 and config.VIS_POS:
        vis.pose_disentanglement_plot(x_test, p_test, vae, device)

    # visualise interpolations
    if config.VIS_INT:
        vis.interpolations_plot(
            x_test, np.ones(len(x_test)), vae, device, poses=p_test
        )
