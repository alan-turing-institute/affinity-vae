import datetime
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

FREQ_ACC = 10
FREQ_REC = 10
FREQ_EMB = 10
FREQ_INT = 10
FREQ_DIS = 10
FREQ_POS = 10
FREQ_SIM = 10
FREQ_EVAL = 10
FREQ_STA = 10

# logging settings


date_time_run = datetime.now().strftime("%H_%M_%d_%m_%Y")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
