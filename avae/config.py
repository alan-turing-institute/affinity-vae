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
VIS_POSE_CLASS = []

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
    "tensorboard": False,
    "classifier": "NN",
    "datatype": "mrc",
    "new_out": False,
    "debug": False,
}
