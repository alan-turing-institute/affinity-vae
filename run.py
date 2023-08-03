import logging
import os
import warnings
from datetime import datetime

import click
import yaml

from avae import config
from avae.evaluate import evaluate
from avae.train import train

if not os.path.exists("logs"):
    os.mkdir("logs")
dt_name = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/avae_run_log_" + dt_name + ".log"),
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
    "--vis_con",
    "-vcn",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise per-class confidence metrics",
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
    "--freq_con",
    "-fr",
    type=int,
    default=None,
    help="Frequency at which to visualise per-class confidence metrics ",
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
    freq_con,
    freq_int,
    freq_dis,
    freq_pos,
    freq_acc,
    freq_sim,
    freq_all,
    vis_rec,
    vis_con,
    vis_los,
    vis_emb,
    vis_int,
    vis_dis,
    vis_pos,
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
):

    warnings.simplefilter("ignore", FutureWarning)
    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    local_vars = locals().copy()
    print(local_vars)

    if config_file is not None:
        with open(config_file, "r") as f:
            logging.info("Reading submission configuration file" + config_file)
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as
        print(data.get("gaussian_blur"))

        for key, val in local_vars.items():
            if (
                val is not None
                and isinstance(val, (int, float, bool, str))
                or data.get(key) is None
            ):
                logging.warning(
                    "Command line argument "
                    + key
                    + " is overwriting config file value to: "
                    + str(val)
                )
                data[key] = val
            else:
                logging.info(
                    "Setting "
                    + key
                    + " to config file value: "
                    + str(data[key])
                )
    else:
        # if no config file is provided, use command line arguments
        data = local_vars

    # Check for missing values and set to default values
    for key, val in data.items():
        if (val is None or val == "None") and key != "config_file":
            #  make sure data variables are provided
            if key == "data_path":
                logging.error(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
                raise ValueError(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Please set a value for this variable."
                )
            elif key == "affinity" or key == "classes":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                filename_default = os.path.join(data["datapath"], key + ".csv")

                if os.path.isfile(filename_default):
                    data[key] = filename_default
                else:
                    data[key] = None

                logging.info(
                    "Setting up "
                    + key
                    + " in config file to "
                    + str(data[key])
                )

            elif key == "state":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest state if in evaluation mode."
                )
            elif key == "meta":
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Loading the latest meta if in evaluation mode."
                )
            else:
                # set missing variables to default value
                logging.warning(
                    "No value set for "
                    + key
                    + " in config file or command line arguments. Setting to default value."
                )
                data[key] = config.DEFAULT_RUN_CONFIGS[key]
                logging.info(
                    "Setting " + key + " to default value: " + str(data[key])
                )

    try:
        if data["vis_all"]:
            config.VIS_LOS = True
            config.VIS_ACC = True
            config.VIS_REC = True
            config.VIS_CON = True
            config.VIS_CYC = True
            config.VIS_AFF = True
            config.VIS_EMB = True
            config.VIS_INT = True
            config.VIS_DIS = True
            config.VIS_POS = True
            config.VIS_HIS = True
            config.VIS_SIM = True

        else:
            config.VIS_LOS = data["vis_los"]
            config.VIS_ACC = data["vis_acc"]
            config.VIS_REC = data["vis_rec"]
            config.VIS_CON = data["vis_con"]
            config.VIS_CYC = data["vis_cyc"]
            config.VIS_AFF = data["vis_aff"]
            config.VIS_EMB = data["vis_emb"]
            config.VIS_INT = data["vis_int"]
            config.VIS_DIS = data["vis_dis"]
            config.VIS_POS = data["vis_pos"]
            config.VIS_HIS = data["vis_his"]
            config.VIS_SIM = data["vis_sim"]

        if data["freq_all"] is not None:
            config.FREQ_EVAL = data["freq_all"]
            config.FREQ_STA = data["freq_all"]
            config.FREQ_ACC = data["freq_all"]
            config.FREQ_REC = data["freq_all"]
            config.FREQ_CON = data["freq_all"]
            config.FREQ_EMB = data["freq_all"]
            config.FREQ_INT = data["freq_all"]
            config.FREQ_DIS = data["freq_all"]
            config.FREQ_POS = data["freq_all"]
            config.FREQ_SIM = data["freq_all"]
        else:
            config.FREQ_EVAL = data["freq_eval"]
            config.FREQ_REC = data["freq_rec"]
            config.FREQ_CON = data["freq_con"]
            config.FREQ_EMB = data["freq_emb"]
            config.FREQ_INT = data["freq_int"]
            config.FREQ_DIS = data["freq_dis"]
            config.FREQ_POS = data["freq_pos"]
            config.FREQ_ACC = data["freq_acc"]
            config.FREQ_STA = data["freq_sta"]
            config.FREQ_SIM = data["freq_sim"]

        file = open("avae_final_config" + dt_name + ".yaml", "w")
        yaml.dump(data, file)
        file.close()
        logging.info("YAML File saved!")

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
                collect_meta=data["dynamic"],
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
                collect_meta=data["dynamic"],
                use_gpu=data["gpu"],
                gaussian_blur=data["gaussian_blur"],
                normalise=data["normalise"],
                shift_min=data["shift_min"],
            )
            # TODO also make sure image is correct size, maybe in dataloader?

        logging.info(
            "Saving final submission config file to: "
            + "avae_final_config"
            + dt_name
            + ".yaml"
        )

    except Exception:
        logging.exception("An exception was thrown!")


if __name__ == "__main__":
    run()
