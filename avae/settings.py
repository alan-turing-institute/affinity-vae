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

FREQ_ACC = 0
FREQ_REC = 0
FREQ_EMB = 0
FREQ_INT = 0
FREQ_DIS = 0
FREQ_POS = 0
FREQ_SIM = 0
FREQ_EVAL = 0
FREQ_STA = 0

# logging settings


date_time_run = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
