import logging
import os

import yaml
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    ValidationError,
)


class AffinityConfig(BaseModel):
    affinity: FilePath = Field(None, description="Path to affinity file")
    batch: PositiveInt = Field(128, description="Batch size")
    beta: float = Field(1, description="Beta value")
    beta_cycle: PositiveInt = Field(4, description="Beta cycle")
    beta_load: FilePath = Field(None, description="Path to beta file")
    beta_min: float = Field(0, description="Minimum betvalue")
    beta_ratio: PositiveFloat = Field(0, description="Beta value")
    channels: PositiveInt = Field(64, description="First layer channels")
    classes: FilePath = Field(None, description="Path to classes file")
    classifier: str = Field(
        "NN",
        pattern='^(KNN|NN|LR)$',
        description="Method to classify the latent space. Options "
        "are: KNN (nearest neighbour), NN (neural network), LR (Logistic Regression).",
    )
    config_file: FilePath = Field(None, description="Path to config file")
    cyc_method_beta: str = Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    cyc_method_gamma: str = Field(
        'flat',
        pattern='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    datapath: DirectoryPath = Field(None, description="Path to data directory")
    datatype: str = Field('mrc', pattern='^npy|mrc$', description="Data type")
    debug: bool = Field(False, description="Debug mode")
    depth: PositiveInt = Field(3, description="Number of layers")
    dynamic: bool = Field(False, description="Dynamic visualisation")
    epochs: PositiveInt = Field(20, description="Number of epochs")
    eval: bool = Field(False, description="Evaluation mode")
    freq_acc: PositiveInt = Field(
        10, description="Frequency (in epochs) of accuracy plot"
    )
    freq_all: PositiveInt = Field(
        None, description="Frequency (in epochs) of all plots"
    )
    freq_dis: PositiveInt = Field(
        10, description="Frequency (in epochs) of disentanglement plot"
    )
    freq_emb: PositiveInt = Field(
        10, description="Frequency (in epochs) of embedding plot"
    )
    freq_eval: PositiveInt = Field(
        10, description="Frequency (in epochs) of evaluation"
    )
    freq_int: PositiveInt = Field(
        10, description="Frequency (in epochs) of interpolation plot"
    )
    freq_pos: PositiveInt = Field(
        10, description="Frequency (in epochs) of pose plot"
    )
    freq_rec: PositiveInt = Field(
        10, description="Frequency (in epochs) of reconstruction plot"
    )
    freq_sim: PositiveInt = Field(
        10, description="Frequency (in epochs) of similarity plot"
    )
    freq_sta: PositiveInt = Field(
        10, description="Frequency (in epochs) of states saved."
    )
    gamma: float = Field(2, description="Gamma value")
    gamma_cycle: PositiveInt = Field(4, description="Gamma cycle")
    gamma_load: FilePath = Field(None, description="Path to gamma array file")
    gamma_min: float = Field(0, description="Minimum gamma value")
    gamma_ratio: float = Field(0.5, description="Gamma ratio")
    gaussian_blur: bool = Field(False, description=" Apply gaussian blur")
    gpu: bool = Field(True, description="Use GPU")
    latent_dims: PositiveInt = Field(8, description="Latent space dimensions")
    learning: PositiveFloat = Field(0.001, description="Learning rate")
    limit: PositiveInt = Field(None, description="Limit number of samples")
    loss_fn: str = Field('MSE', description="Loss function")
    meta: FilePath = Field(None, description="Path to meta file")
    model: str = Field('a', description="Type of model to use")
    new_out: bool = Field(False, description="Create new output directory")
    no_val_drop: bool = Field(
        True,
        description="Do not drop last validation batch if is smaller than batch size",
    )
    normalise: bool = Field(False, description="Normalise data")
    opt_method: str = Field(
        'adam',
        description="Optimisation method.It can be adam/sgd/asgd",
        pattern='^(adam|sgd|asgd)$',
    )
    pose_dims: PositiveInt = Field(1, description="Pose dimensions")
    rescale: bool = Field(False, description="Rescale data")
    restart: bool = Field(False, description="Restart training")
    shift_min: bool = Field(
        False, description="Scale data with min-max transformation"
    )
    split: PositiveInt = Field(20, description="Split ratio")
    state: FilePath = Field(None, description="Path to state file")
    tensorboard: bool = Field(False, description="Use tensorboard")
    vis_acc: bool = Field(False, description="Visualise accuracy")
    vis_aff: bool = Field(False, description="Visualise affinity")
    vis_all: bool = Field(False, description="Visualise all")
    vis_cyc: bool = Field(False, description="Visualise beta/gamma cycle")
    vis_dis: bool = Field(False, description="Visualise disentanglement")
    vis_emb: bool = Field(False, description="Visualise embedding")
    vis_his: bool = Field(False, description="Visualise history")
    vis_int: bool = Field(False, description="Visualise interpolation")
    vis_los: bool = Field(False, description="Visualise loss")
    vis_pos: bool = Field(False, description="Visualise pose")
    vis_pose_class: str = Field(
        False, description="Visualise pose classification"
    )
    vis_rec: bool = Field(False, description="Visualise reconstruction")
    vis_sim: bool = Field(False, description="Visualise similarity")


def load_config_params(config_file, local_vars):

    if config_file is not None:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f)

        try:
            data = AffinityConfig(**config_data)
            print("Config file is valid!")
        except ValidationError as e:
            print("Config file is invalid:")
            print(e)
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
                data.key = val
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

    return data


def write_config_file(time_stamp_name, data):
    # record final configuration in logger and save to yaml file
    for key, val in data.items():
        logging.info("Parameter " + key + " set to value: " + str(data[key]))

    if not os.path.exists("configs"):
        os.mkdir("configs")
    file = open("configs/avae_final_config" + time_stamp_name + ".yaml", "w")
    yaml.dump(data, file)
    file.close()

    logging.info("YAML File saved!\n")


VIS_LOS = False
VIS_ACC = False
VIS_REC = False
VIS_EMB = False
VIS_INT = False
VIS_DIS = False
VIS_POS = False
VIS_CYC = False
VIS_AFF = False
VIS_HIS = False
VIS_SIM = False
VIS_DYN = False
VIS_POSE_CLASS = False

FREQ_ACC = 10
FREQ_REC = 10
FREQ_EMB = 10
FREQ_INT = 10
FREQ_DIS = 10
FREQ_POS = 10
FREQ_SIM = 10

FREQ_EVAL = 10
FREQ_STA = 10

DEFAULT_RUN_CONFIGS = {
    "limit": None,
    "restart": False,
    "split": 20,
    "depth": 3,
    "channels": 64,
    "latent_dims": 8,
    "pose_dims": 1,
    "no_val_drop": True,
    "epochs": 20,
    "batch": 128,
    "learning": 0.001,
    "gpu": True,
    "beta": 1.0,
    "gamma": 2,
    "beta_load": None,
    "gamma_load": None,
    "loss_fn": "MSE",
    "beta_min": 0.0,
    "beta_cycle": 4,
    "beta_ratio": 0.5,
    "cyc_method_beta": "flat",
    "gamma_min": 0.0,
    "gamma_cycle": 4,
    "gamma_ratio": 0.5,
    "cyc_method_gamma": "flat",
    "freq_eval": FREQ_EVAL,
    "freq_sta": FREQ_STA,
    "freq_emb": FREQ_EMB,
    "freq_rec": FREQ_REC,
    "freq_int": FREQ_INT,
    "freq_dis": FREQ_DIS,
    "freq_pos": FREQ_POS,
    "freq_acc": FREQ_ACC,
    "freq_sim": FREQ_SIM,
    "freq_all": None,
    "eval": False,
    "dynamic": VIS_DYN,
    "vis_los": VIS_LOS,
    "vis_acc": VIS_ACC,
    "vis_rec": VIS_REC,
    "vis_emb": VIS_EMB,
    "vis_int": VIS_INT,
    "vis_dis": VIS_DIS,
    "vis_pos": VIS_POS,
    "vis_pose_class": VIS_POSE_CLASS,
    "vis_cyc": VIS_CYC,
    "vis_aff": VIS_AFF,
    "vis_his": VIS_HIS,
    "vis_sim": VIS_SIM,
    "vis_all": False,
    "model": "a",
    "opt_method": "adam",
    "gaussian_blur": False,
    "normalise": False,
    "shift_min": False,
    "rescale": False,
    "tensorboard": False,
    "classifier": "NN",
    "datatype": "mrc",
    "new_out": False,
    "debug": False,
}
