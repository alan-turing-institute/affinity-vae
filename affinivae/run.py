import click
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Subset

from model import VariationalAutoencoder
from data import ProteinDataset
from train import run_train, run_validate
from evaluate import run_evaluate
from vis import merge, vis_latentembed_plot, vis_loss_plot, vis_recon_plot, vis_interp_grid, \
    vis_single_transversals, vis_single_transversals_pose
# from similarity import create_lookup_tensor, create_scores_df


@click.command(name="Pokemino Trainer")
@click.option('--datapath', '-d', type=str, default=None, required=True, help="Path to training data.")
@click.option('--limit', '-lm', type=int, default=None,
              help="Limit the number of samples loaded (default None).")
@click.option('--split', '-sp', type=int, default=10, help="Train/val split in %.")
@click.option('--epochs', '-ep', type=int, default=100, help="Number of epochs (default 100).")
@click.option('--batch', '-ba', type=int, default=128, help="Batch size (default 128).")
@click.option('--learning', '-lr', type=float, default=1e-4, help="Learning rate (default 1e-4).")
@click.option('--depth', '-de', type=int, default=3, help="Depth of the convolutional layers (default 3).")
@click.option('--channels', '-ch', type=int, default=64, help="First layer channels (default 64).")
@click.option('--latent_dims', '-ld', type=int, default=10, help="Latent space dimension (default 10).")
@click.option('--loss_fn', '-lf', type=str, default='MSE', help="Loss type: 'MSE' or 'BCE' (default 'MSE').")
@click.option('--beta', '-b', type=float, default=1.0, help="Variational beta (default 1).")
@click.option('--pose', '-ps', type=bool, default=False, is_flag=True, help='Turn pose channel on.')
@click.option('--pose_dims', '-pd', type=int, default=1, help='If pose on, number of pose dimensions.')
@click.option('--gamma', '-g', type=float, default=1.0, help="Scale factor for the loss component corresponding "
                                                             "to shape similarity (default 1).")
@click.option('--freq_eval', '-fev', type=int, default=10, help="Frequency at which to evaluate test set "
                                                                "(default every 10 epochs).")
@click.option('--freq_sta', '-fs', type=int, default=10, help="Frequency at which to save state "
                                                              "(default every 10 epochs).")
@click.option('--freq_lat', '-fs', type=int, default=10, help="Frequency at which to visualise the latent space "
                                                              "(default every 10 epochs).")
@click.option('--freq_emb', '-fe', type=int, default=10, help="Frequency at which to visualise the latent "
                                                              "space embedding (default every 10 epochs).")
@click.option('--freq_rec', '-fr', type=int, default=10, help="Frequency at which to visualise reconstructions "
                                                              "(default every 10 epochs).")
@click.option('--freq_int', '-fi', type=int, default=10, help="Frequency at which to visualise latent space"
                                                              "interpolations (default every 10 epochs).")
@click.option('--freq_dis', '-ft', type=int, default=10, help="Frequency at which to visualise single transversals "
              "(default every 10 epochs).")
@click.option('--freq_pos', '-fp', type=int, default=10, help="Frequency at which to visualise pose "
              "(default every 10 epochs).")
@click.option('--freq_acc', '-fac', type=int, default=10, help="Frequency at which to visualise confusion matrix.")
@click.option('--freq_all', '-fa', type=int, default=None, help="Frequency at which to visualise all plots except loss "
              "(default every 10 epochs).")
@click.option('--vis_lat', '-vs', type=bool, default=False, is_flag=True,
              help="Visualise latent space (or it's first two dimensions).")
@click.option('--vis_emb', '-ve', type=bool, default=False, is_flag=True,
              help="Visualise latent space embedding.")
@click.option('--vis_rec', '-vr', type=bool, default=False, is_flag=True,
              help="Visualise reconstructions.")
@click.option('--vis_los', '-vl', type=bool, default=False, is_flag=True, help="Visualise loss.")
@click.option('--vis_int', '-vi', type=bool, default=False, is_flag=True, help="Visualise interpolations.")
@click.option('--vis_dis', '-vt', type=bool, default=False, is_flag=True, help="Visualise single transversals.")
@click.option('--vis_pos', '-vps', type=bool, default=False, is_flag=True, help="Visualise pose interpolations in the "
                                                                                "first 2 dimensions")
@click.option('--vis_acc', '-vac', type=bool, default=False, is_flag=True, help="Visualise confusion matrix.")
@click.option('--vis_all', '-va', type=bool, default=False, is_flag=True, help="Visualise all above.")
@click.option('--gpu', '-g', type=bool, default=False, is_flag=True, help="Use GPU for training.")
@click.option('--evaluate', '-ev', type=bool, default=False, is_flag=True, help="Evaluate test data.")
@click.option('--no_val_drop', '-nd', type=bool, default=False, is_flag=True, help="Do not drop last validate batch if "
                                                                                   "if it is smaller than batch_size.")
def run(datapath, limit, split, epochs, batch, learning,
        depth, channels, latent_dims, loss_fn, beta,
        pose, pose_dims,  gamma,
        freq_eval, freq_sta, freq_lat, freq_emb, freq_rec, freq_int, freq_dis, freq_pos, freq_acc, freq_all,
        vis_lat, vis_emb, vis_rec, vis_los, vis_int, vis_dis, vis_pos, vis_acc, vis_all,
        gpu, evaluate, no_val_drop):

    print()
    torch.manual_seed(42)
    if vis_all:
        vis_los, vis_lat, vis_emb, vis_rec, vis_int, vis_dis, vis_pos, vis_acc = [True] * 8
    if freq_all:
        freq_eval, freq_lat, freq_emb, freq_rec, freq_int, freq_dis, freq_pos, freq_acc = [freq_all] * 8

    # ############################### DATA ###############################
    lookup = [f for f in os.listdir(datapath) if 'scores' in f]
    if len(lookup) > 1:
        raise RuntimeError("More than 1 affinity matrix in the root directory {}.".format(datapath))
    elif not (len(lookup) == 0 and 'test' in datapath):
        lookup = lookup[0]
    lookup = pd.read_csv(os.path.join(datapath, lookup)).set_index('Unnamed: 0')

    if not evaluate:
        # create ProteinDataset
        data = ProteinDataset(datapath, lookup, lim=limit)
        print("Data size:", len(data))

        # split into train / val sets
        idx = np.random.permutation(len(data))
        s = int(np.ceil(len(data) * int(split)/100))
        if s < 2:
            raise RuntimeError("Train and validation sets must be larger than 1 sample, train: {}, val: {}.".format(
                len(idx[:-s]), len(idx[-s:])))
        train_data = Subset(data, indices=idx[:-s])
        val_data = Subset(data, indices=idx[-s:])
        print("Train / val split:", len(train_data), len(val_data))

        # split into batches
        trains = DataLoader(train_data, batch_size=batch, num_workers=8, shuffle=True, drop_last=True)
        vals = DataLoader(val_data, batch_size=batch, num_workers=8, shuffle=True, drop_last=(not no_val_drop))
        if len(vals) < 1 or len(trains) < 1:
            # ensure the batch size is not smaller than validation set
            raise RuntimeError("Validation or train set is too small for the current batch size. Please edit either "
                               "split percent '-sp/--split' or batch size '-ba/--batch' or set "
                               "'-nd/--no_val_drop flag' (only if val is too small). Batch: {}, train:{}, val: {}, "
                               "split: {}%.".format(batch, len(train_data), len(val_data), split))
        print("Train / val batches:", len(trains), len(vals))

    if evaluate or ('test' in os.listdir(datapath)):
        data = ProteinDataset(os.path.join(datapath, 'test'), lookup, lim=limit)
        print("Eval data size:", len(data))
        tests = DataLoader(data, batch_size=batch, num_workers=8, shuffle=True)
        print("Eval batches:", len(tests))
    dsize = data[0]['img'].shape[-3:]

    lookup = lookup.to_numpy(dtype=np.float32)
    print()

    # ############################### MODEL ###############################
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
    if gpu and device == "cpu":
        print("\nWARNING: no GPU available, running on CPU instead.\n")

    if not evaluate:
        vae = VariationalAutoencoder(channel_init=channels, latent_dims=latent_dims, depth=depth,
                                     input_size=dsize,
                                     lookup=lookup, gamma=gamma, pose=pose,
                                     pose_dims=pose_dims)
        optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning)  # , weight_decay=1e-5)
        train_loss, recon_loss, kldiv_loss, affin_loss, val_loss = [], [], [], [], []
    else:
        vae = torch.load('avae.pt')
    vae.to(device)
    # print(vae)

    for epoch in range(epochs):

        meta_df = pd.DataFrame()

        train_loss.append(0)
        recon_loss.append(0)
        kldiv_loss.append(0)
        affin_loss.append(0)
        val_loss.append(0)

        if not evaluate:
            # ############################### TRAIN ###############################
            x, x_hat, meta_df, train_loss, recon_loss, kldiv_loss, affin_loss = \
                run_train(vae, optimizer, device, trains,
                          latent_dims, pose_dims, beta, gamma, loss_fn, epoch, epochs,
                          train_loss, recon_loss, kldiv_loss, affin_loss, meta_df,
                          vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
                          freq_emb, freq_int, freq_pos, freq_dis, freq_acc)

            # ############################### VALIDATE ###############################
            x_val, x_hat_val, meta_df, val_loss = run_validate(vae, device, vals,
                                             latent_dims, pose_dims, beta, gamma, loss_fn, epoch, epochs,
                                             val_loss, meta_df,
                                             vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
                                             freq_emb, freq_int, freq_pos, freq_dis, freq_acc)

        # ############################### EVALUATE ###############################
        if evaluate or ('test' in os.listdir(datapath) and (epoch + 1) % freq_eval == 0):
            x, x_hat, meta_df = run_evaluate(vae, device, tests, latent_dims, meta_df)

        # ############################### VISUALISE ###############################
        # save state
        if (epoch+1) % freq_sta == 0:
            torch.save(vae, 'avae.pt')

        # visualise loss
        if not evaluate:
            if vis_los and epoch > 0:
                vis_loss_plot(epoch+1, train_loss, recon_loss=recon_loss, kldiv_loss=kldiv_loss, shape_loss=affin_loss, val_loss = val_loss,
                              p=[len(trains), depth, channels, latent_dims, learning, beta, gamma])

        # visualise reconstructions
        if vis_rec and (epoch+1) % freq_rec == 0:
            vis_recon_plot(x, x_hat)
            if not evaluate:
                vis_recon_plot(x_val, x_hat_val, val=True)

        # visualise embeddings
        if (vis_emb and (epoch+1) % freq_emb == 0) or evaluate:
            meta_df['image'] = meta_df['image'].apply(merge)    # merge img and rec into one image for display in altair
            vis_latentembed_plot(meta_df, epoch, embedding='umap')
            vis_latentembed_plot(meta_df, epoch, embedding='tsne')

        # visualise latent disentanglement
        if vis_dis and (epoch + 1) % freq_dis == 0:
            vis_single_transversals(meta_df, vae, device, dsize, latent_dims, pose_dims,
                                    pose=pose)

        # visualise pose disentanglement
        if pose and vis_pos and (epoch + 1) % freq_pos == 0:
            vis_single_transversals_pose(meta_df, vae, device, dsize, latent_dims, pose_dims)

        # visualise interpolations
        if vis_int and (epoch+1) % freq_int == 0:
            vis_interp_grid(meta_df, vae, device, dsize, pose=pose)

        # ############################### WRAP-UP ###############################

        # empty meta_df at each epoch
        meta_df = pd.DataFrame()

        # only one epoch if evaluating
        if evaluate:
            exit(0)


if __name__ == '__main__':
    run()
