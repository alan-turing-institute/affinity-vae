import numpy as np
import pandas as pd

from vis import format


def run_evaluate(vae, device, tests, latent_dims, meta_df):
    vae.eval()
    for b, batch in enumerate(tests):
        # forward
        x = batch['img'].to(device)
        x_hat, latent_mu, latent_logvar, lat, lat_pose = vae(x)
        print('EVAL Batch: [%d/%d]' % (b + 1, len(tests)), end='\r')

        # save metadata
        s = pd.DataFrame(batch['meta'])
        s['mode'] = 'test'
        s['image'] += format(x_hat)
        for d in range(latent_dims):
            s[f"lat{d}"] = np.array(latent_mu[:, d].cpu().detach().numpy())
        meta_df = pd.concat([meta_df, s], ignore_index=False)  # ignore index doesn't overwrite

    print('EVAL Batch: [%d/%d]' % (b + 1, len(tests)))

    return x, x_hat, meta_df


if __name__ is "__main__":
    # TODO write evaluate only routine
    pass
