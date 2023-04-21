import logging
from datetime import datetime

import click
import yaml

from avae import config
from avae.evaluate import evaluate
from avae.train import train

dt_name = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("avae_run_log_" + dt_name + ".log"),
        logging.StreamHandler(),
    ],
)


@click.command(name="Affinity Trainer")
@click.option("--config_file", type=click.Path(exists=True))
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
    "--no_val_drop",
    "-nd",
    type=bool,
    default=False,
    is_flag=True,
    help="Do not drop last validate batch if "
    "if it is smaller than batch_size.",
)
@click.option(
    "--affinity",
    "-af",
    type=str,
    default=None,
    help="Path to affinity matrix for training.",
)
@click.option(
    "--classes",
    "-cl",
    type=str,
    default=None,
    help="Path to a CSV file containing a list of classes for training.",
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
    "--pose_dims",
    "-pd",
    type=int,
    default=0,
    help="If pose on, number of pose dimensions. If 0 and gamma=0 it becomes"
    "a standard beta-VAE.",
)
@click.option(
    "--beta",
    "-b",
    type=float,
    default=1.0,
    help="Variational beta (default 1).",
)
@click.option(
    "--gamma",
    "-g",
    type=float,
    default=1.0,
    help="Scale factor for the loss component corresponding "
    "to shape similarity (default 1). If 0 and pd=0 it becomes a standard"
    "beta-VAE.",
)
@click.option(
    "--learning",
    "-lr",
    type=float,
    default=1e-4,
    help="Learning rate (default 1e-4).",
)
@click.option(
    "--loss_fn",
    "-lf",
    type=str,
    default="MSE",
    help="Loss type: 'MSE' or 'BCE' (default 'MSE').",
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
    "--eval",
    "-ev",
    type=bool,
    default=False,
    is_flag=True,
    help="Evaluate test data.",
)
@click.option(
    "--dynamic",
    "-dn",
    type=bool,
    default=False,
    is_flag=True,
    help="Enable collecting meta and dynamic latent space plots.",
)
def run(
    datapath,
    limit,
    split,
    no_val_drop,
    affinity,
    classes,
    epochs,
    batch,
    depth,
    channels,
    latent_dims,
    pose_dims,
    beta,
    gamma,
    learning,
    loss_fn,
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
    eval,
    dynamic,
):

    print()
    if vis_all:
        config.VIS_LOS = True
        config.VIS_EMB = True
        config.VIS_REC = True
        config.VIS_INT = True
        config.VIS_DIS = True
        config.VIS_POS = True
        config.VIS_ACC = True
    else:
        config.VIS_LOS = vis_los
        config.VIS_EMB = vis_emb
        config.VIS_REC = vis_rec
        config.VIS_INT = vis_int
        config.VIS_DIS = vis_dis
        config.VIS_POS = vis_pos
        config.VIS_ACC = vis_acc

    if freq_all is not None:
        config.FREQ_EVAL = freq_all
        config.FREQ_EMB = freq_all
        config.FREQ_REC = freq_all
        config.FREQ_INT = freq_all
        config.FREQ_DIS = freq_all
        config.FREQ_POS = freq_all
        config.FREQ_ACC = freq_all
        config.FREQ_STA = freq_all
    else:
        config.FREQ_EVAL = freq_eval
        config.FREQ_EMB = freq_emb
        config.FREQ_REC = freq_rec
        config.FREQ_INT = freq_int
        config.FREQ_DIS = freq_dis
        config.FREQ_POS = freq_pos
        config.FREQ_ACC = freq_acc
        config.FREQ_STA = freq_sta

    if not eval:
        train(
            datapath,
            limit,
            split,
            batch,
            no_val_drop,
            affinity,
            classes,
            dynamic,
            epochs,
            channels,
            depth,
            latent_dims,
            pose_dims,
            learning,
            beta,
            gamma,
            loss_fn,
            gpu,
        )
    else:
        evaluate(datapath, limit, split, batch, dynamic, gpu)
        # TODO also make sure image is correct size, maybe in dataloader?


if __name__ == "__main__":
    run()
