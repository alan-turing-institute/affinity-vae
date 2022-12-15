import numpy as np
import pandas as pd

from vis import vis_accuracy, format


def run_train(vae, optimizer, device, trains,
              latent_dims, pose_dims, beta, gamma, loss_fn, epoch, epochs,
              train_loss, recon_loss, kldiv_loss, affin_loss, meta_df,
              vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
              freq_emb, freq_int, freq_pos, freq_dis, freq_acc):
    vae.train()

    for b, batch in enumerate(trains):
        # forward
        x = batch['img'].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)
        loss, rloss, kloss, aloss = vae.loss(
            x_hat, x, latent_mu, latent_logvar, beta,
            gamma=gamma, ids=batch['aff'], lats=lat,
            loss_fn=loss_fn, device=device, epoch=epoch)

        # record loss
        train_loss[-1] += loss.item()
        recon_loss[-1] += rloss.item()
        kldiv_loss[-1] += kloss.item()
        affin_loss[-1] += aloss.item()
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
            s = pd.DataFrame(batch['meta'])
            s['mode'] = 'train'
            s['image'] += format(x_hat)
            for d in range(latent_dims):
                s[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
            for d in range(pose_dims):
                s[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
            meta_df = pd.concat([meta_df, s], ignore_index=False)  # ignore index doesn't overwrite

    # print loss
    train_loss[-1] /= len(trains)
    recon_loss[-1] /= len(trains)
    kldiv_loss[-1] /= len(trains)
    affin_loss[-1] /= len(trains)
    print('Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f | Recon: %f | KLdiv: %f | Affin: %f' % (
        epoch + 1, epochs,
        b + 1, len(trains),
        train_loss[-1],
        recon_loss[-1],
        kldiv_loss[-1],
        affin_loss[-1]))

    return x, x_hat, meta_df, train_loss, recon_loss, kldiv_loss, affin_loss


def run_validate(vae, device, vals,
                 latent_dims, pose_dims, beta, gamma, loss_fn, epoch, epochs,
                 val_loss, meta_df,
                 vis_emb, vis_int, vis_pos, vis_dis, vis_acc,
                 freq_emb, freq_int, freq_pos, freq_dis, freq_acc):

    vae.eval()
    for b, batch in enumerate(vals):

        # forward
        x = batch['img'].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)

        # record loss
        vloss = vae.loss(
            x_hat, x, latent_mu, latent_logvar, beta,
            gamma=gamma, ids=batch['aff'], lats=lat,
            loss_fn=loss_fn, device=device, epoch=epoch)
        val_loss[-1] += vloss[0].item()
        print('VAL Epoch: [%d/%d] | Batch: [%d/%d] | Loss: %f' % (
            epoch + 1, epochs, b + 1, len(vals), vloss[0]), end='\r')

        # save metadata
        if (vis_emb and (epoch + 1) % freq_emb == 0) or \
                (vis_dis and (epoch + 1) % freq_dis == 0) or \
                (vis_int and (epoch + 1) % freq_int == 0) or \
                (vis_pos and (epoch + 1) % freq_pos == 0) or \
                (vis_acc and (epoch + 1) % freq_acc == 0):
            s = pd.DataFrame(batch['meta'])
            s['mode'] = 'val'
            s['image'] += format(x_hat)
            for d in range(latent_dims):
                s[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
            for d in range(pose_dims):
                s[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
            meta_df = pd.concat([meta_df, s], ignore_index=False)  # ignore index doesn't overwrite

    val_loss[-1] /= len(vals)

    # accuracy
    if (epoch + 1) % freq_acc != 0:
        print('VAL: [%d/%d] | Batch: [%d/%d] | Loss: %f' % (
            epoch + 1, epochs, b + 1, len(vals), val_loss[-1]))
    else:
        acc = vis_accuracy(meta_df)
        print('VAL: [%d/%d] | Batch: [%d/%d] | Loss: %f | Train_acc: %f | Val_acc: %f' % (
            epoch + 1, epochs, b + 1, len(vals), val_loss[-1], acc[0], acc[1]))

    return x, x_hat, meta_df, val_loss


if __name__ is "__main__":
    # TODO write train / val only routine
    pass