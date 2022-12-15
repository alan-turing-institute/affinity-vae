import numpy as np
import pandas as pd

from vis import format


def run_train(epoch, epochs, vae, optimizer, beta, gamma, loss_fn, device, trains, meta_df,
              vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
              freq_emb, freq_int, freq_pos, freq_dis, freq_acc):
    """Defines a single epoch training pass.

    Parameters
    ----------
    epoch : int
        Current epoch.
    epochs : int
        Total number of epochs.
    vae : torch.nn.Module
        Model.
    optimizer : torch.optim
        Optimizer.
    beta: float
        Beta parameter defining weight on latent regularisation term.
    gamma : float
        Gamma parameter defining weight on affinity regularisation term. If gamma is None, affinity loss
        is not computed and does not count towards the total loss.
    loss_fn : 'MSE' or 'BCE'
        Function used for reconstruction loss. BCE uses Binary Cross-Entropy for binary data and MSE uses Mean
        Squared Error for real-valued data.
    device : torch.device
        Device to train on, e.g. GPU or CPU.
    trains : torch.utils.data.DataLoader
        A batched Dataset.

    Visualisation Parameters TODO rewrite this as a dict of vis params
    ---------------
    vis_emb : bool
        Saves metadata for embedding visualisations.
    vis_int : bool
        Saves metadata for interpolation visualisations.
    vis_pos : bool
        Saves metadata for pose interpolation visualisations.
    vis_dis : bool
        Saves metadata for disentanglement visualisation.
    vis_acc : bool
        Saves metadata for confusion matrix visualisation.

    freq_emb : int
        Defines frequency of embedding visualisations (in epochs).
    freq_int : int
        Defines frequency of interpolation visualisations (in epochs).
    freq_pos : int
        Defines frequency of pose interpolation visualisations (in epochs).
    freq_dis : int
        Defines frequency of disentanglement visualisations (in epochs).
    freq_acc : int
        Defines frequency of confusion visualisations (in epochs).

    Returns
    -------
    x : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of inputs, where N stands for the number of samples in the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    x_hat : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    meta : dict
        Associated metadata.
    train_loss : int
        Average total loss across the epoch.
    recon_loss : int
        Average reconstruction loss across the epoch.
    kldiv_loss : int
         Average latent regularisation loss across the epoch.
     affin_loss : int
        Average affine regularisation loss across the epoch.
    """
    vae.train()

    train_loss, recon_loss, kldiv_loss, affin_loss = 0, 0, 0, 0

    for b, batch in enumerate(trains):
        # forward
        x = batch['img'].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)
        loss, rloss, kloss, aloss = vae.loss(
            x_hat, x, latent_mu, latent_logvar, beta,
            gamma=gamma, ids=batch['aff'],
            loss_fn=loss_fn, device=device)

        # record loss
        train_loss += loss.item()
        recon_loss += rloss.item()
        kldiv_loss += kloss.item()
        affin_loss += aloss.item()
        print('Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | KLdiv: %f | Affin: %f' % (
            epoch + 1, epochs,
            b + 1, len(trains),
            loss,
            rloss,
            kloss,
            aloss), end='\r')

        # backpropagate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # save metadata for plots
        if (vis_emb and (epoch + 1) % freq_emb == 0) or \
                (vis_dis and (epoch + 1) % freq_dis == 0) or \
                (vis_int and (epoch + 1) % freq_int == 0) or \
                (vis_pos and (epoch + 1) % freq_pos == 0) or \
                (vis_acc and (epoch + 1) % freq_acc == 0):
            meta = pd.DataFrame(batch['meta'])
            meta['mode'] = 'train'
            meta['image'] += format(x_hat)
            for d in range(latent_mu.shape[-1]):
                meta[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
            for d in range(lat_pose.shape[-1]):
                meta[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
            meta_df = pd.concat([meta_df, meta], ignore_index=False)  # ignore index doesn't overwrite

    # print loss
    train_loss /= len(trains)
    recon_loss /= len(trains)
    kldiv_loss /= len(trains)
    affin_loss /= len(trains)
    print('Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | KLdiv: %f | Affin: %f' % (
        epoch + 1, epochs,
        b + 1, len(trains),
        train_loss, recon_loss, kldiv_loss, affin_loss))

    return x, x_hat, meta_df, train_loss, recon_loss, kldiv_loss, affin_loss


def run_validate(epoch, epochs, vae, beta, gamma, loss_fn, device, vals, meta_df,
                 vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
                 freq_emb, freq_int, freq_pos, freq_dis, freq_acc):
    """Defines a single epoch validation pass.

    Parameters
    ----------
    epoch : int
        Current epoch.
    epochs : int
        Total number of epochs.
    vae : torch.nn.Module
        Model.
    beta: float
        Beta parameter defining weight on latent regularisation term.
    gamma : float
        Gamma parameter defining weight on affinity regularisation term. If gamma is None, affinity loss
        is not computed and does not count towards the total loss.
    loss_fn : 'MSE' or 'BCE'
        Function used for reconstruction loss. BCE uses Binary Cross-Entropy for binary data and MSE uses Mean
        Squared Error for real-valued data.
    device : torch.device
        Device to train on, e.g. GPU or CPU.
    vals : torch.utils.data.DataLoader
        A batched Dataset.

    Visualisation Parameters
    ---------------
    vis_emb : bool
        Saves metadata for embedding visualisations.
    vis_int : bool
        Saves metadata for interpolation visualisations.
    vis_pos : bool
        Saves metadata for pose interpolation visualisations.
    vis_dis : bool
        Saves metadata for disentanglement visualisation.
    vis_acc : bool
        Saves metadata for confusion matrix visualisation.

    freq_emb : int
        Defines frequency of embedding visualisations (in epochs).
    freq_int : int
        Defines frequency of interpolation visualisations (in epochs).
    freq_pos : int
        Defines frequency of pose interpolation visualisations (in epochs).
    freq_dis : int
        Defines frequency of disentanglement visualisations (in epochs).
    freq_acc : int
        Defines frequency of confusion visualisations (in epochs).

    Returns
    -------
    x : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of inputs, where N stands for the number of samples in the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    x_hat : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    meta : dict
        Associated metadata.
    val_loss : int
        Average total loss across the epoch.
    """
    vae.eval()

    val_loss = 0

    for b, batch in enumerate(vals):

        # forward
        x = batch['img'].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)

        # record loss
        vloss = vae.loss(
            x_hat, x, latent_mu, latent_logvar, beta,
            gamma=gamma, ids=batch['aff'],
            loss_fn=loss_fn, device=device)
        val_loss += vloss[0].item()
        print('VAL Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f' % (
            epoch + 1, epochs, b + 1, len(vals), vloss[0]), end='\r')

        # save metadata TODO make metadata collection optional
        if (vis_emb and (epoch + 1) % freq_emb == 0) or \
                (vis_dis and (epoch + 1) % freq_dis == 0) or \
                (vis_int and (epoch + 1) % freq_int == 0) or \
                (vis_pos and (epoch + 1) % freq_pos == 0) or \
                (vis_acc and (epoch + 1) % freq_acc == 0):
            meta = pd.DataFrame(batch['meta'])
            meta['mode'] = 'val'
            meta['image'] += format(x_hat)
            for d in range(latent_mu.shape[-1]):
                meta[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
            for d in range(lat_pose.shape[-1]):
                meta[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
            meta_df = pd.concat([meta_df, meta], ignore_index=False)  # ignore index doesn't overwrite


    val_loss /= len(vals)

    # accuracy
    print('VAL: [%d/%d] | Batch: [%d/%d] | Loss: %f' % (
        epoch + 1, epochs, b + 1, len(vals), val_loss))

    return x, x_hat, meta_df, val_loss


if __name__ is "__main__":
    # TODO write train / val only routine
    pass
