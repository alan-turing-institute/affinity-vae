import yaml
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    Float,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    conbool,
    confloat,
)


class AffinityConfig(BaseModel):
    affinity: FilePath = None
    batch: PositiveInt = Field(128, description="Batch size")
    beta: Float = Field(1, description="Beta value")
    beta_cycle: PositiveInt = Field(4, description="Beta cycle")
    beta_load: FilePath = None
    beta_min: Float = Field(0, description="Minimum betvalue")
    beta_ratio: Float = Field(0, description="Beta value")
    channels: PositiveInt = Field(64, description="First layer channels")
    classes: FilePath = None
    classifier: str = Field(
        "NN",
        regex='^(KNN|NN|LR)$',
        description="Method to classify the latent space. Options "
        "are: KNN (nearest neighbour), NN (neural network), LR (Logistic Regression).",
    )
    config_file: FilePath = None
    cyc_method_beta: str = Field(
        'flat',
        regex='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    cyc_method_gamma: str = Field(
        'flat',
        regex='^(cycle_sigmoid|flat|cycle_linear|cycle_cosine|ramp)$',
    )
    datapath: DirectoryPath = None
    datatype: str = Field(regex='^npy|$')
    debug: conbool()
    depth: PositiveInt
    dynamic: conbool()
    epochs: PositiveInt
    eval: conbool()
    freq_acc: PositiveInt
    freq_all: PositiveInt
    freq_dis: PositiveInt
    freq_emb: PositiveInt
    freq_eval: PositiveInt
    freq_int: PositiveInt
    freq_pos: PositiveInt
    freq_rec: PositiveInt
    freq_sim: PositiveInt
    freq_sta: PositiveInt
    gamma: confloat(ge=0)
    gamma_cycle: PositiveInt
    gamma_load: FilePath = None
    gamma_min: NonNegativeInt
    gamma_ratio: confloat(ge=0, le=1)
    gaussian_blur: conbool()
    gpu: conbool()
    latent_dims: PositiveInt
    learning: confloat(ge=0)
    limit: PositiveInt
    loss_fn: constr(min_length=1)
    meta: type(None)
    model: constr(min_length=1)
    new_out: conbool()
    no_val_drop: conbool()
    normalise: conbool()
    opt_method: constr(min_length=1)
    pose_dims: PositiveInt
    rescale: conbool()
    restart: conbool()
    shift_min: conbool()
    split: PositiveInt
    state: type(None)
    tensorboard: conbool()
    vis_acc: conbool()
    vis_aff: conbool()
    vis_all: conbool()
    vis_cyc: conbool()
    vis_dis: conbool()
    vis_emb: conbool()
    vis_his: conbool()
    vis_int: conbool()
    vis_los: conbool()
    vis_pos: conbool()
    vis_pose_class: conlist(conint(ge=0), min_items=1)
    vis_rec: conbool()
    vis_sim: conbool()


def load_config_params(config_file, local_vars):

    if config_file is not None:
        with open(config_file, "r") as f:
            config_data = yaml.safe_load(f, Loader=yaml.FullLoader)

        try:
            data = AppConfig(**config_data)
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
