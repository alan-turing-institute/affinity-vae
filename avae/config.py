VIS_LOS = False
VIS_EMB = False
VIS_REC = False
VIS_INT = False
VIS_DIS = False
VIS_POS = False
VIS_ACC = False
VIS_HIS = False

FREQ_EVAL = False
FREQ_EMB = False
FREQ_REC = False
FREQ_INT = False
FREQ_DIS = False
FREQ_POS = False
FREQ_ACC = False
FREQ_STA = False

DEFAULT_RUN_CONFIGS = {
    "limit": 1000,
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
    "beta": 1,
    "gamma": 2,
    "loss_fn": "MSE",
    "freq_eval": FREQ_EVAL,
    "freq_sta": FREQ_STA,
    "freq_emb": FREQ_EMB,
    "freq_rec": FREQ_REC,
    "freq_int": FREQ_INT,
    "freq_dis": FREQ_DIS,
    "freq_pos": FREQ_POS,
    "freq_acc": FREQ_ACC,
    "freq_all": False,
    "eval": False,
    "dynamic": False,
    "vis_emb": VIS_EMB,
    "vis_rec": VIS_REC,
    "vis_los": VIS_LOS,
    "vis_int": VIS_INT,
    "vis_dis": VIS_DIS,
    "vis_pos": VIS_POS,
    "vis_acc": VIS_ACC,
    "vis_his": VIS_HIS,
    "vis_all": False,
    "model": "a",
}
