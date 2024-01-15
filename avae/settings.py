import logging
from datetime import datetime

## visualisation configurations

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
VIS_POSE_CLASS = None
VIS_Z_N_INT = None

FREQ_ACC = None
FREQ_REC = None
FREQ_EMB = None
FREQ_INT = None
FREQ_DIS = None
FREQ_POS = None
FREQ_SIM = None
FREQ_EVAL = None
FREQ_STA = None

# logging settings


date_time_run = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
