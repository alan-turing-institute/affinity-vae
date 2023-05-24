import os

import numpy as np
import pandas as pd
import torch

from . import config, vis
from .data import load_data
from .train import add_meta, pass_batch
from .utils import set_device


def evaluate(datapath, state, lim, splt, batch_s, collect_meta, use_gpu):

    # ############################### DATA ###############################
    tests = load_data(datapath, lim, splt, batch_s, collect_meta, eval=True)

    # ############################### MODEL ###############################
    device = set_device(use_gpu)

    if state == None and not os.path.exists("states"):
        raise RuntimeError(
            "There are no existing model states saved, unable to evaluate."
        )
    # TODO add param to chose model
    if state == None : 
        state = sorted([s for s in os.listdir("states") if ".pt" in s])[0]
        fname = state.split(".")[0].split("_")
        pose_dims = fname[3]
        vae = torch.load(os.path.join("states", state))  # make optional param


    vae.to(device)

    # ########################## EVALUATE ################################
    if collect_meta:
        metas = sorted([f for f in os.listdir("states") if ".pkl" in f])[-1]
        meta_df = pd.read_pickle(metas)

    # create holders for latent spaces and labels
    x_test = []
    if pose_dims != 0:
        p_test = []

    print("Batch: [0/%d]" % (len(tests)), end="\r", flush=True)

    vae.eval()
    for b, batch in enumerate(tests):
        x, x_hat, lat_mu, lat_logvar, lat, lat_pose, _ = pass_batch(
            device, vae, batch, b, len(tests)
        )
        x_test.extend(lat_mu.cpu().detach().numpy())  # store latents
        if pose_dims != 0:
            p_test.extend(lat_pose.cpu().detach().numpy())

        if collect_meta:  # store meta for plots
            meta_df = add_meta(
                meta_df, batch[-1], x_hat, lat_mu, lat_pose, mode="evl"
            )

        print("Batch: [%d/%d]" % (b + 1, len(tests)), end="\r", flush=True)
    print("Batch: [%d/%d]" % (b + 1, len(tests)), flush=True)

    # ########################## VISUALISE ################################

    # visualise reconstructions - last batch
    if config.VIS_REC:
        vis.recon_plot(x, x_hat, name="evl")

    # visualise embeddings
    if config.VIS_EMB:
        vis.latent_embed_plot(x_test, np.ones(len(x_test)))
        if collect_meta:
            # merge img and rec into one image for display in altair
            meta_df["image"] = meta_df["image"].apply(vis.merge)
            vis.dyn_latentembed_plot(meta_df, 0, embedding="umap")
            vis.dyn_latentembed_plot(meta_df, 0, embedding="tsne")

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
