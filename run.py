import logging
import os
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
    "--gamma",
    "-g",
    type=float,
    default=None,
    help="Scale factor for the loss component corresponding "
    "to shape similarity. If 0 and pd=0 it becomes a standard"
    "beta-VAE.",
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
    "--kl_weight_method",
    "-klm",
    type=str,
    default=None,
    help="The schedule for beta: for constant beta : flat, other options include , cycle_linear, cycle_sigmoid, cycle_cosine, ramp",
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
    "--freq_emb",
    "-fe",
    type=int,
    default=None,
    help="Frequency at which to visualise the latent " "space embedding.",
)
@click.option(
    "--freq_rec",
    "-fr",
    type=int,
    default=None,
    help="Frequency at which to visualise reconstructions ",
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
    "--freq_acc",
    "-fac",
    type=int,
    default=None,
    help="Frequency at which to visualise confusion matrix.",
)
@click.option(
    "--freq_all",
    "-fa",
    type=int,
    default=None,
    help="Frequency at which to visualise all plots except loss. ",
)
@click.option(
    "--vis_emb",
    "-ve",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise latent space embedding.",
)
@click.option(
    "--vis_rec",
    "-vr",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise reconstructions.",
)
@click.option(
    "--vis_los",
    "-vl",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise loss.",
)
@click.option(
    "--vis_bet",
    "-vb",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise beta vs epoch number.",
)
@click.option(
    "--vis_int",
    "-vi",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise interpolations.",
)
@click.option(
    "--vis_dis",
    "-vt",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise single transversals.",
)
@click.option(
    "--vis_pos",
    "-vps",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise pose interpolations in the first 2 dimensions",
)
@click.option(
    "--vis_acc",
    "-vac",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise confusion matrix.",
)
@click.option(
    "--vis_his",
    "-his",
    type=bool,
    default=None,
    is_flag=True,
    help="Visualise train-val class distribution.",
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
def run(
    config_file,
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
    beta_min,
    beta_cycle,
    beta_ratio,
    kl_weight_method,
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
    vis_bet,
    vis_los,
    vis_int,
    vis_dis,
    vis_pos,
    vis_acc,
    vis_his,
    vis_all,
    gpu,
    eval,
    dynamic,
    model,
):
    # read config file and command line arguments and assign to local variables that are used in the rest of the code
    local_vars = locals().copy()
    if config_file is not None:
        with open(config_file, "r") as f:
            logging.info("Reading submission configuration file" + config_file)
            data = yaml.load(f, Loader=yaml.FullLoader)
        # returns JSON object as
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
            config.VIS_BET = True
            config.VIS_LOS = True
            config.VIS_EMB = True
            config.VIS_REC = True
            config.VIS_INT = True
            config.VIS_DIS = True
            config.VIS_POS = True
            config.VIS_ACC = True
            config.VIS_HIS = True

        else:
            config.VIS_BET = data["vis_bet"]
            config.VIS_LOS = data["vis_los"]
            config.VIS_EMB = data["vis_emb"]
            config.VIS_REC = data["vis_rec"]
            config.VIS_INT = data["vis_int"]
            config.VIS_DIS = data["vis_dis"]
            config.VIS_POS = data["vis_pos"]
            config.VIS_ACC = data["vis_acc"]
            config.VIS_HIS = data["vis_his"]

        if data["freq_all"] is not None:
            config.FREQ_EVAL = data["freq_all"]
            config.FREQ_EMB = data["freq_all"]
            config.FREQ_REC = data["freq_all"]
            config.FREQ_INT = data["freq_all"]
            config.FREQ_DIS = data["freq_all"]
            config.FREQ_POS = data["freq_all"]
            config.FREQ_ACC = data["freq_all"]
            config.FREQ_STA = data["freq_all"]
        else:
            config.FREQ_EVAL = data["freq_eval"]
            config.FREQ_EMB = data["freq_emb"]
            config.FREQ_REC = data["freq_rec"]
            config.FREQ_INT = data["freq_int"]
            config.FREQ_DIS = data["freq_dis"]
            config.FREQ_POS = data["freq_pos"]
            config.FREQ_ACC = data["freq_acc"]
            config.FREQ_STA = data["freq_sta"]

        if not data["eval"]:
            train(
                data["datapath"],
                data["limit"],
                data["split"],
                data["batch"],
                data["no_val_drop"],
                data["affinity"],
                data["classes"],
                data["dynamic"],
                data["epochs"],
                data["channels"],
                data["depth"],
                data["latent_dims"],
                data["pose_dims"],
                data["learning"],
                data["beta_min"],
                data["beta"],
                data["beta_cycle"],
                data["beta_ratio"],
                data["kl_weight_method"],
                data["gamma"],
                data["loss_fn"],
                data["gpu"],
                data["model"],
            )
        else:
            evaluate(
                data["datapath"],
                data["limit"],
                data["split"],
                data["batch"],
                data["dynamic"],
                data["gpu"],
            )
            # TODO also make sure image is correct size, maybe in dataloader?

        logging.info(
            "Saving final submission config file to: "
            + "avae_final_config"
            + dt_name
            + ".yaml"
        )
        file = open("avae_final_config" + dt_name + ".yaml", "w")
        yaml.dump(data, file)
        file.close()
        logging.info("YAML File saved!")

    except Exception:
        logging.exception("An exception was thrown!")


if __name__ == "__main__":
    run()
