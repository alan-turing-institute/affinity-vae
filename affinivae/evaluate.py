import numpy as np
import pandas as pd
from vis import format


def run_evaluate(vae, device, tests, meta_df):
    """Defines a single epoch validation pass.

    Parameters
    ----------
    vae : torch.nn.Module
        Model.
    device : torch.device
        Device to train on, e.g. GPU or CPU.
    tests : torch.utils.data.DataLoader
        A batched Dataset.

    Returns
    -------
    x : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of inputs, where N stands for the number of samples in
        the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    x_hat : torch.Tensor (N, CH, Z, Y, X)
        Last mini-batch of outputs, where N stands for the number of samples in the mini-batch, CH stands for number of
        channels and X, Y, Z define input dimensions.
    meta : dict
        Associated metadata.
    """
    vae.eval()
    for b, batch in enumerate(tests):
        # forward
        x = batch["img"].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)
        print("EVAL Batch: [%d/%d]" % (b + 1, len(tests)), end="\r")

        # save metadata
        meta = pd.DataFrame(batch["meta"])
        meta["mode"] = "test"
        meta["image"] += format(x_hat)
        for d in range(latent_mu.shape[-1]):
            meta[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
        for d in range(lat_pose.shape[-1]):
            meta[f"pos{d}"] = np.array(lat_pose[:, d].cpu().detach().numpy())
        meta_df = pd.concat(
            [meta_df, meta], ignore_index=False
        )  # ignore index doesn't overwrite

    print("EVAL Batch: [%d/%d]" % (b + 1, len(tests)))

    return x, x_hat, meta_df


if __name__ == "__main__":
    # TODO write evaluate only routine
    pass
