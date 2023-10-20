import logging
import os
import warnings
from datetime import datetime

import click

from avae import config
from avae.evaluate import evaluate
from avae.train import train
from avae.utils import load_config_params, write_config_file

dt_name = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@click.command(name="Affinity Trainer")
@click.option("--config_file", type=click.Path(exists=True))
@click.option(
    "--datapath",
    "-d",
    type=str,
    default=None,
    help="Path to training data.",
)
@click.option(
    "--datatype",
    "-dtype",
    type=str,
    default=None,
    help="Type of the data: mrc, npy",
)
@click.option(
    "--debug",
    "-dbg",
    type=bool,
    default=None,
    is_flag=True,
    help="Run in debug mode.",
)
@click.option(
    "--restart",
    "-res",
    type=bool,
    default=None,
    is_flag=True,
    help="Is the calculation restarting from a checkpoint.",
)
@click.option(
    "--state",
    "-st",
    type=str,
    default=None,
    help="The saved model state to be loaded for evaluation/resume training.",
)
@click.option(
    "--meta",
    "-mt",
    type=str,
    default=None,
    help="The saved meta file to be loaded for regenerating dynamic plots.",
)
@click.option(
    "--limit",
    "-lm",
    type=int,
    default=None,
    help="Limit the number of samples loaded (default None).",
)
@click.option(
    "--split", "-sp", type=int, default=None, help="Train/val split in %."
)
@click.option(
    "--new_out",
    "-newo",
    type=bool,
    default=None,
    is_flag=True,
    help="Create new output directory where to save the results.",
)
@click.option(
    "--no_val_drop",
    "-nd",
    type=bool,
    default=None,
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
    "--classifier",
    "-clf",
    type=str,
    default=None,
    help="Method to classify the latent space. Options are: KNN (nearest neighbour), NN (neural network), LR (Logistic Regression).",
)
@click.option(
    "--epochs",
    "-ep",
    type=int,
    default=None,
    help="Number of epochs (default 100).",
)
@click.option(
    "--batch", "-ba", type=int, default=None, help="Batch size (default 128)."
)
@click.option(
    "--depth",
    "-de",
    type=int,
    default=None,
    help="Depth of the convolutional layers (default 3).",
)
@click.option(
    "--channels",
    "-ch",
    type=int,
    default=None,
    help="First layer channels (default 64).",
)
@click.option(
    "--latent_dims",
    "-ld",
    type=int,
    default=None,
    help="Latent space dimension (default 10).",
)
@click.option(
    "--pose_dims",
    "-pd",
    type=int,
    default=None,
    help="If pose on, number of pose dimensions. If 0 and gamma=0 it becomes"
    "a standard beta-VAE.",
)
@click.option(
    "--beta",
    "-be",
    type=float,
    default=None,
    help="Beta maximum in the case of cyclical annealing schedule",
)
@click.option(
    "--beta_load",
    "-bl",
    type=str,
    default=None,
    is_flag=True,
    help="The path to the saved beta array file to be loaded "
    "if this file is provided, all other beta related variables would be ignored",
)
@click.option(
    "--gamma",
    "-g",
    type=float,
    default=None,
    help="Scale factor for the loss component corresponding "
    "to shape similarity. If 0 and pd=0 it becomes a standard"
    "beta-VAE.",
)
@click.option(
    "--gamma_load",
    "-gl",
    type=str,
    default=None,
    is_flag=True,
    help="The path to the saved gamma array file to be loaded"
    "if this file is provided, all other gamma related variables would be ignored",
)
@click.option(
    "--learning",
    "-lr",
    type=float,
    default=None,
    help="Learning rate.",
)
@click.option(
    "--loss_fn",
    "-lf",
    type=str,
    default=None,
    help="Loss type: 'MSE' or 'BCE' (default 'MSE').",
)
@click.option(
    "--beta_min",
    "-bs",
    type=float,
    default=None,
    help="Beta minimum in the case of cyclical annealing schedule",
)
@click.option(
    "--beta_cycle",
    "-bc",
    type=int,
    default=None,
    help="Number of cycles for beta during training in the case of cyclical annealing schedule",
)
@click.option(
    "--beta_ratio",
    "-br",
    type=float,
    default=None,
    help="The ratio for steps in beta",
)
@click.option(
    "--cyc_method_beta",
    "-cycmb",
    type=str,
    default=None,
    help="The schedule for : for constant beta : flat, other options include , cycle_linear, cycle_sigmoid, cycle_cosine, ramp",
)
@click.option(
    "--gamma_min",
    "-gs",
    type=float,
    default=None,
    help="gamma minimum in the case of cyclical annealing schedule",
)
@click.option(
    "--gamma_cycle",
    "-gc",
    type=int,
    default=None,
    help="Number of cycles for gamma during training in the case of cyclical annealing schedule",
)
@click.option(
    "--gamma_ratio",
    "-gr",
    type=float,
    default=None,
    help="The ratio for steps in gamma",
)
@click.option(
    "--cyc_method_gamma",
    "-cycmg",
    type=str,
    default=None,
    help="The schedule for gamma: for constant gamma : flat, other options include , cycle_linear, cycle_sigmoid, cycle_cosine, ramp",
)
@click.option(
    "--gpu",
    "-g",
    type=bool,
    default=None,
    is_flag=True,
    help="Use GPU for training.",
)
@click.option(
    "--eval",
    "-ev",
    type=bool,
    default=None,
    is_flag=True,
    help="Evaluate test data.",
)
@click.option(
    "--dynamic",
    "-dn",
    type=bool,
    default=None,
    is_flag=True,
    help="Enable collecting meta and dynamic latent space plots.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default=None,
    help="Choose model to run.",
)
@click.option(
    "--vis_los",
    "-vl",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise loss (every epoch starting at epoch 2).",
)
@click.option(
    "--vis_acc",
    "-vac",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise confusion matrix and F1 scores (frequency controlled).",
)
@click.option(
    "--vis_rec",
    "-vr",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise reconstructions (frequency controlled).",
)
@click.option(
    "--vis_emb",
    "-ve",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise latent space embedding (frequency controlled).",
)
@click.option(
    "--vis_int",
    "-vi",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise interpolations (frequency controlled).",
)
@click.option(
    "--vis_dis",
    "-vt",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise latent disentanglement (frequency controlled).",
)
@click.option(
    "--vis_pos",
    "-vps",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise pose disentanglement (frequency controlled).",
)
@click.option(
    "--vis_pose_class",
    "-vpsc",
    type=str,
    default=None,
    help="Example: A,B,C. your deliminator should be commas and no spaces .Classes to be used for pose interpolation (a seperate pose interpolation figure would be created for each class).",
)
@click.option(
    "--vis_cyc",
    "-vc",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise cyclical parameters (once per run).",
)
@click.option(
    "--vis_aff",
    "-va",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise affinity matrix (once per run).",
)
@click.option(
    "--vis_his",
    "-his",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise train-val class distribution (once per run).",
)
@click.option(
    "--vis_sim",
    "-similarity",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise train-val model similarity matrix.",
)
@click.option(
    "--vis_all",
    "-va",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise all above.",
)
@click.option(
    "--freq_eval",
    "-fev",
    type=int,
    default=None,
    help="Frequency at which to evaluate test set.",
)
@click.option(
    "--freq_sta",
    "-fs",
    type=int,
    default=None,
    help="Frequency at which to save state",
)
@click.option(
    "--freq_acc",
    "-fac",
    type=int,
    default=None,
    help="Frequency at which to visualise confusion matrix.",
)
@click.option(
    "--freq_rec",
    "-fr",
    type=int,
    default=None,
    help="Frequency at which to visualise reconstructions ",
)
@click.option(
    "--freq_emb",
    "-fe",
    type=int,
    default=None,
    help="Frequency at which to visualise the latent " "space embedding.",
)
@click.option(
    "--freq_int",
    "-fi",
    type=int,
    default=None,
    help="Frequency at which to visualise latent space"
    "interpolations (default every 10 epochs).",
)
@click.option(
    "--freq_dis",
    "-ft",
    type=int,
    default=None,
    help="Frequency at which to visualise single transversals.",
)
@click.option(
    "--freq_pos",
    "-fp",
    type=int,
    default=None,
    help="Frequency at which to visualise pose.",
)
@click.option(
    "--freq_sim",
    "-fsim",
    type=int,
    default=None,
    help="Frequency at which to visualise similarity matrix.",
)
@click.option(
    "--freq_all",
    "-fa",
    type=int,
    default=None,
    help="Frequency at which to visualise all plots except loss. ",
)
@click.option(
    "--opt_method",
    "-opt",
    type=str,
    default=None,
    help=" The method of optimisation. It can be adam/sgd/asgd",
)
@click.option(
    "--gaussian_blur",
    "-gb",
    type=bool,
    default=None,
    is_flag=True,
    help="Applying gaussian bluring to the image data which should help removing noise. The minimum and maximum for this is hardcoded.",
)
@click.option(
    "--normalise",
    "-nrm",
    type=bool,
    default=None,
    is_flag=True,
    help="Normalise data",
)
@click.option(
    "--shift_min",
    "-sftm",
    type=bool,
    default=None,
    is_flag=True,
    help="Shift the minimum of the data to one zero and the maximum to one",
)
@click.option(
    "--rescale",
    "-res",
    type=int,
    default=None,
    is_flag=False,
    help="Rescale images to given value (tuple, one value per dim).",
)
@click.option(
    "--tensorboard",
    "-tb",
    type=bool,
    default=None,
    is_flag=True,
    help="Log metrics and figures to tensorboard during training",
)
def run(
    config_file,
    datapath,
    datatype,
    restart,
    state,
    meta,
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
    beta_load,
    gamma_load,
    gamma,
    learning,
    loss_fn,
    beta_min,
    beta_cycle,
    beta_ratio,
    cyc_method_beta,
    gamma_min,
    gamma_cycle,
    gamma_ratio,
    cyc_method_gamma,
    freq_eval,
    freq_sta,
    freq_emb,
    freq_rec,
    freq_int,
    freq_dis,
    freq_pos,
    freq_acc,
    freq_sim,
    freq_all,
    vis_rec,
    vis_los,
    vis_emb,
    vis_int,
    vis_dis,
    vis_pos,
    vis_pose_class,
    vis_acc,
    vis_cyc,
    vis_aff,
    vis_his,
    vis_sim,
    vis_all,
    gpu,
    eval,
    dynamic,
    model,
    opt_method,
    gaussian_blur,
    normalise,
    shift_min,
    rescale,
    tensorboard,
    classifier,
    new_out,
    debug,
):

    warnings.simplefilter("ignore", FutureWarning)

    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    logging.info("Reading submission configuration file" + config_file)
    local_vars = locals().copy()
    data = load_config_params(config_file, local_vars)

    if data["debug"]:
        logging.info("Debug mode enabled")
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("matplotlib.font_manager").disabled = True

    if data["vis_all"]:
        config.VIS_LOS = True
        config.VIS_ACC = True
        config.VIS_REC = True
        config.VIS_CYC = True
        config.VIS_AFF = True
        config.VIS_EMB = True
        config.VIS_INT = True
        config.VIS_DIS = True
        config.VIS_POS = True
        config.VIS_HIS = True
        config.VIS_SIM = True
        config.VIS_DYN = True
        config.VIS_POSE_CLASS = data["vis_pose_class"]

    else:
        config.VIS_LOS = data["vis_los"]
        config.VIS_ACC = data["vis_acc"]
        config.VIS_REC = data["vis_rec"]
        config.VIS_CYC = data["vis_cyc"]
        config.VIS_AFF = data["vis_aff"]
        config.VIS_EMB = data["vis_emb"]
        config.VIS_INT = data["vis_int"]
        config.VIS_DIS = data["vis_dis"]
        config.VIS_POS = data["vis_pos"]
        config.VIS_HIS = data["vis_his"]
        config.VIS_SIM = data["vis_sim"]
        config.VIS_DYN = data["dynamic"]
        config.VIS_POSE_CLASS = data["vis_pose_class"]

    if data["freq_all"] is not None:
        config.FREQ_EVAL = data["freq_all"]
        config.FREQ_STA = data["freq_all"]
        config.FREQ_ACC = data["freq_all"]
        config.FREQ_REC = data["freq_all"]
        config.FREQ_EMB = data["freq_all"]
        config.FREQ_INT = data["freq_all"]
        config.FREQ_DIS = data["freq_all"]
        config.FREQ_POS = data["freq_all"]
        config.FREQ_SIM = data["freq_all"]
    else:
        config.FREQ_EVAL = data["freq_eval"]
        config.FREQ_REC = data["freq_rec"]
        config.FREQ_EMB = data["freq_emb"]
        config.FREQ_INT = data["freq_int"]
        config.FREQ_DIS = data["freq_dis"]
        config.FREQ_POS = data["freq_pos"]
        config.FREQ_ACC = data["freq_acc"]
        config.FREQ_STA = data["freq_sta"]
        config.FREQ_SIM = data["freq_sim"]

    if data["new_out"]:
        dir_name = f'results_{dt_name}_lat{data["latent_dims"]}_pose{data["pose_dims"]}_lr{data["learning"]}_beta{data["beta"]}_gamma{data["gamma"]}'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        os.chdir(dir_name)

    if not os.path.exists("logs"):
        os.mkdir("logs")

    # setup logger inside the  directory where we are running the code
    fileh = logging.FileHandler("logs/avae_run_log_" + dt_name + ".log", "a")
    logging.getLogger().addHandler(fileh)

    logging.info(
        "Saving final submission config file to: "
        + "avae_final_config"
        + dt_name
        + ".yaml"
    )

    write_config_file(dt_name, data)

    try:
        run_pipeline(data)

    except Exception as e:
        logging.exception("An exception was thrown!", e)


def run_pipeline(data):

    if not data["eval"]:
        train(
            datapath=data["datapath"],
            datatype=data["datatype"],
            restart=data["restart"],
            state=data["state"],
            lim=data["limit"],
            splt=data["split"],
            batch_s=data["batch"],
            no_val_drop=data["no_val_drop"],
            affinity=data["affinity"],
            classes=data["classes"],
            epochs=data["epochs"],
            channels=data["channels"],
            depth=data["depth"],
            lat_dims=data["latent_dims"],
            pose_dims=data["pose_dims"],
            learning=data["learning"],
            beta_load=data["beta_load"],
            beta_min=data["beta_min"],
            beta_max=data["beta"],
            beta_cycle=data["beta_cycle"],
            beta_ratio=data["beta_ratio"],
            cyc_method_beta=data["cyc_method_beta"],
            gamma_load=data["gamma_load"],
            gamma_min=data["gamma_min"],
            gamma_max=data["gamma"],
            gamma_cycle=data["gamma_cycle"],
            gamma_ratio=data["gamma_ratio"],
            cyc_method_gamma=data["cyc_method_gamma"],
            recon_fn=data["loss_fn"],
            use_gpu=data["gpu"],
            model=data["model"],
            opt_method=data["opt_method"],
            gaussian_blur=data["gaussian_blur"],
            normalise=data["normalise"],
            shift_min=data["shift_min"],
            rescale=data["rescale"],
            tensorboard=data["tensorboard"],
            classifier=data["classifier"],
        )
    else:
        evaluate(
            datapath=data["datapath"],
            datatype=data["datatype"],
            state=data["state"],
            meta=data["meta"],
            lim=data["limit"],
            splt=data["split"],
            batch_s=data["batch"],
            classes=data["classes"],
            use_gpu=data["gpu"],
            gaussian_blur=data["gaussian_blur"],
            normalise=data["normalise"],
            shift_min=data["shift_min"],
            rescale=data["rescale"],
            classifier=data["classifier"],
        )


if __name__ == "__main__":
    run()
