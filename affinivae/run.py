import os

import click
import numpy as np
import pandas as pd
import torch
from data import ProteinDataset
from evaluate import run_evaluate
from model_b import AffinityVAE
from torch.utils.data import DataLoader, Subset
from train import run_train, run_validate
from vis import (
    merge,
    vis_accuracy,
    vis_interp_grid,
    vis_latentembed_plot,
    vis_loss_plot,
    vis_recon_plot,
    vis_single_transversals,
    vis_single_transversals_pose,
)


@click.command(name="Affinity Trainer")
@click.option(
    "--datapath",
    "-d",
    type=str,
    default=None,
    required=True,
    help="Path to training data.",
)
@click.option(
    "--limit",
    "-lm",
    type=int,
    default=None,
    help="Limit the number of samples loaded (default None).",
)
@click.option(
    "--split", "-sp", type=int, default=10, help="Train/val split in %."
)
@click.option(
    "--epochs",
    "-ep",
    type=int,
    default=100,
    help="Number of epochs (default 100).",
)
@click.option(
    "--batch", "-ba", type=int, default=128, help="Batch size (default 128)."
)
@click.option(
    "--learning",
    "-lr",
    type=float,
    default=1e-4,
    help="Learning rate (default 1e-4).",
)
@click.option(
    "--depth",
    "-de",
    type=int,
    default=3,
    help="Depth of the convolutional layers (default 3).",
)
@click.option(
    "--channels",
    "-ch",
    type=int,
    default=64,
    help="First layer channels (default 64).",
)
@click.option(
    "--latent_dims",
    "-ld",
    type=int,
    default=10,
    help="Latent space dimension (default 10).",
)
@click.option(
    "--loss_fn",
    "-lf",
    type=str,
    default="MSE",
    help="Loss type: 'MSE' or 'BCE' (default 'MSE').",
)
@click.option(
    "--beta",
    "-b",
    type=float,
    default=1.0,
    help="Variational beta (default 1).",
)
@click.option(
    "--pose",
    "-ps",
    type=bool,
    default=False,
    is_flag=True,
    help="Turn pose channel on.",
)
@click.option(
    "--pose_dims",
    "-pd",
    type=int,
    default=1,
    help="If pose on, number of pose dimensions.",
)
@click.option(
    "--gamma",
    "-g",
    type=float,
    default=1.0,
    help="Scale factor for the loss component corresponding "
    "to shape similarity (default 1).",
)
@click.option(
    "--freq_eval",
    "-fev",
    type=int,
    default=10,
    help="Frequency at which to evaluate test set "
    "(default every 10 epochs).",
)
@click.option(
    "--freq_sta",
    "-fs",
    type=int,
    default=10,
    help="Frequency at which to save state " "(default every 10 epochs).",
)
@click.option(
    "--freq_emb",
    "-fe",
    type=int,
    default=10,
    help="Frequency at which to visualise the latent "
    "space embedding (default every 10 epochs).",
)
@click.option(
    "--freq_rec",
    "-fr",
    type=int,
    default=10,
    help="Frequency at which to visualise reconstructions "
    "(default every 10 epochs).",
)
@click.option(
    "--freq_int",
    "-fi",
    type=int,
    default=10,
    help="Frequency at which to visualise latent space"
    "interpolations (default every 10 epochs).",
)
@click.option(
    "--freq_dis",
    "-ft",
    type=int,
    default=10,
    help="Frequency at which to visualise single transversals "
    "(default every 10 epochs).",
)
@click.option(
    "--freq_pos",
    "-fp",
    type=int,
    default=10,
    help="Frequency at which to visualise pose " "(default every 10 epochs).",
)
@click.option(
    "--freq_acc",
    "-fac",
    type=int,
    default=10,
    help="Frequency at which to visualise confusion matrix.",
)
@click.option(
    "--freq_all",
    "-fa",
    type=int,
    default=None,
    help="Frequency at which to visualise all plots except loss "
    "(default every 10 epochs).",
)
@click.option(
    "--vis_emb",
    "-ve",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise latent space embedding.",
)
@click.option(
    "--vis_rec",
    "-vr",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise reconstructions.",
)
@click.option(
    "--vis_los",
    "-vl",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise loss.",
)
@click.option(
    "--vis_int",
    "-vi",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise interpolations.",
)
@click.option(
    "--vis_dis",
    "-vt",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise single transversals.",
)
@click.option(
    "--vis_pos",
    "-vps",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise pose interpolations in the " "first 2 dimensions",
)
@click.option(
    "--vis_acc",
    "-vac",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise confusion matrix.",
)
@click.option(
    "--vis_all",
    "-va",
    type=bool,
    default=False,
    is_flag=True,
    help="Visualise all above.",
)
@click.option(
    "--gpu",
    "-g",
    type=bool,
    default=False,
    is_flag=True,
    help="Use GPU for training.",
)
@click.option(
    "--evaluate",
    "-ev",
    type=bool,
    default=False,
    is_flag=True,
    help="Evaluate test data.",
)
@click.option(
    "--no_val_drop",
    "-nd",
    type=bool,
    default=False,
    is_flag=True,
    help="Do not drop last validate batch if "
    "if it is smaller than batch_size.",
)
def run(
    datapath,
    limit,
    split,
    epochs,
    batch,
    learning,
    depth,
    channels,
    latent_dims,
    loss_fn,
    beta,
    pose,
    pose_dims,
    gamma,
    freq_eval,
    freq_sta,
    freq_emb,
    freq_rec,
    freq_int,
    freq_dis,
    freq_pos,
    freq_acc,
    freq_all,
    vis_emb,
    vis_rec,
    vis_los,
    vis_int,
    vis_dis,
    vis_pos,
    vis_acc,
    vis_all,
    gpu,
    evaluate,
    no_val_drop,
):

    print()
    torch.manual_seed(42)
    if vis_all:
        vis_los, vis_emb, vis_rec, vis_int, vis_dis, vis_pos, vis_acc = [
            True
        ] * 7
    if freq_all:
        (
            freq_eval,
            freq_emb,
            freq_rec,
            freq_int,
            freq_dis,
            freq_pos,
            freq_acc,
            freq_sta,
        ) = [freq_all] * 8

    # ############################### DATA ###############################
    if not evaluate:
        lookup = [f for f in os.listdir(datapath) if "scores" in f]
        if len(lookup) > 1:
            raise RuntimeError(
                "More than 1 affinity matrix in the root directory {}.".format(
                    datapath
                )
            )
        elif not (len(lookup) == 0 and "test" in datapath):
            lookup = lookup[0]
        lookup = pd.read_csv(os.path.join(datapath, lookup)).set_index(
            "Unnamed: 0"
        )

        # create ProteinDataset
        data = ProteinDataset(datapath, amatrix=lookup, lim=limit)
        print("Data size:", len(data))

        # split into train / val sets
        idx = np.random.permutation(len(data))
        s = int(np.ceil(len(data) * int(split) / 100))
        if s < 2:
            raise RuntimeError(
                "Train and validation sets must be larger than 1 sample, train: {}, val: {}.".format(
                    len(idx[:-s]), len(idx[-s:])
                )
            )
        train_data = Subset(data, indices=idx[:-s])
        val_data = Subset(data, indices=idx[-s:])
        print("Train / val split:", len(train_data), len(val_data))

        # split into batches
        trains = DataLoader(
            train_data,
            batch_size=batch,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )
        vals = DataLoader(
            val_data,
            batch_size=batch,
            num_workers=8,
            shuffle=True,
            drop_last=(not no_val_drop),
        )
        if len(vals) < 1 or len(trains) < 1:
            # ensure the batch size is not smaller than validation set
            raise RuntimeError(
                "Validation or train set is too small for the current batch size. Please edit either "
                "split percent '-sp/--split' or batch size '-ba/--batch' or set "
                "'-nd/--no_val_drop flag' (only if val is too small). Batch: {}, train:{}, val: {}, "
                "split: {}%.".format(
                    batch, len(train_data), len(val_data), split
                )
            )
        print("Train / val batches:", len(trains), len(vals))

        lookup = lookup.to_numpy(dtype=np.float32)

    if evaluate or ("test" in os.listdir(datapath)):
        data = ProteinDataset(
            os.path.join(datapath, "test"), amatrix=None, lim=limit
        )
        print("Eval data size:", len(data))
        tests = DataLoader(data, batch_size=batch, num_workers=8, shuffle=True)
        print("Eval batches:", len(tests))

    dsize = data[0]["img"].shape[-3:]

    print()

    # ############################### MODEL ###############################
    device = torch.device(
        "cuda:0" if gpu and torch.cuda.is_available() else "cpu"
    )
    if gpu and device == "cpu":
        print("\nWARNING: no GPU available, running on CPU instead.\n")

    if not evaluate:
        vae = AffinityVAE(
            channels,
            depth,
            dsize,
            latent_dims,
            lookup,
            pose=pose,
            pose_dims=pose_dims,
        )
        optimizer = torch.optim.Adam(
            params=vae.parameters(), lr=learning
        )  # , weight_decay=1e-5)
        train_loss, recon_loss, kldiv_loss, affin_loss, val_loss = (
            [],
            [],
            [],
            [],
            [],
        )
    else:
        vae = torch.load("avae.pt")
    vae.to(device)

    for epoch in range(epochs):

        meta_df = pd.DataFrame()

        if not evaluate:
            # ############################### TRAIN ###############################
            x, x_hat, meta_df, tloss, rloss, kloss, aloss = run_train(
                epoch,
                epochs,
                vae,
                optimizer,
                beta,
                gamma,
                loss_fn,
                device,
                trains,
                meta_df,
                vis_emb,
                vis_int,
                vis_pos,
                vis_dis,
                vis_acc,
                freq_emb,
                freq_int,
                freq_pos,
                freq_dis,
                freq_acc,
            )

            train_loss.append(tloss)
            recon_loss.append(rloss)
            kldiv_loss.append(kloss)
            affin_loss.append(aloss)

            # ############################### VALIDATE ###############################
            x_val, x_hat_val, meta_df, vloss = run_validate(
                epoch,
                epochs,
                vae,
                beta,
                gamma,
                loss_fn,
                device,
                vals,
                meta_df,
                vis_emb,
                vis_int,
                vis_pos,
                vis_dis,
                vis_acc,
                freq_emb,
                freq_int,
                freq_pos,
                freq_dis,
                freq_acc,
            )
            val_loss.append(vloss)

        # ############################### EVALUATE ###############################
        if evaluate or (
            "test" in os.listdir(datapath) and (epoch + 1) % freq_eval == 0
        ):
            x, x_hat, meta_df = run_evaluate(vae, device, tests, meta_df)

        # ############################### VISUALISE ###############################
        # save state
        if (epoch + 1) % freq_sta == 0:
            torch.save(vae, "avae.pt")

        # visualise accuracy
        if vis_acc and (epoch + 1) % freq_acc == 0:
            acc = vis_accuracy(meta_df)
            print(
                "------------------->>> Accuracy: Train: %f | Val: %f\n"
                % (acc[0], acc[1])
            )

        # visualise loss
        if not evaluate:
            if vis_los and epoch > 0:
                vis_loss_plot(
                    epoch + 1,
                    train_loss,
                    recon_loss=recon_loss,
                    kldiv_loss=kldiv_loss,
                    shape_loss=affin_loss,
                    val_loss=val_loss,
                    p=[
                        len(trains),
                        depth,
                        channels,
                        latent_dims,
                        learning,
                        beta,
                        gamma,
                    ],
                )

        # visualise reconstructions
        if vis_rec and (epoch + 1) % freq_rec == 0:
            vis_recon_plot(x, x_hat)
            if not evaluate:
                vis_recon_plot(x_val, x_hat_val, val=True)

        # visualise embeddings
        if (vis_emb and (epoch + 1) % freq_emb == 0) or evaluate:
            meta_df["image"] = meta_df["image"].apply(
                merge
            )  # merge img and rec into one image for display in altair
            vis_latentembed_plot(meta_df, epoch, embedding="umap")
            vis_latentembed_plot(meta_df, epoch, embedding="tsne")

        # visualise latent disentanglement
        if vis_dis and (epoch + 1) % freq_dis == 0:
            vis_single_transversals(
                meta_df, vae, device, dsize, latent_dims, pose_dims, pose=pose
            )

        # visualise pose disentanglement
        if pose and vis_pos and (epoch + 1) % freq_pos == 0:
            vis_single_transversals_pose(
                meta_df, vae, device, dsize, latent_dims, pose_dims
            )

        # visualise interpolations
        if vis_int and (epoch + 1) % freq_int == 0:
            vis_interp_grid(meta_df, vae, device, dsize, pose=pose)

        # ############################### WRAP-UP ###############################

        # only one epoch if evaluating
        if evaluate:
            exit(0)


if __name__ == "__main__":
    run()
