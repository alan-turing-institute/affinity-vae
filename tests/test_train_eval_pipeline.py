import os
import random
import shutil
import tempfile
import unittest

import torch

from avae import config
from run import run_pipeline
from tests import testdata_mrc, testdata_npy

# fixing random seeds so we dont get fail on mrc tests
torch.random.manual_seed(10)
random.seed(10)


class TrainEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        """Test instantiation of the pipeline."""

        self.testdata_mrc = os.path.dirname(testdata_mrc.__file__)
        self.testdata_npy = os.path.dirname(testdata_npy.__file__)

        self.data = {
            "datapath": self.testdata_mrc,
            "datatype": "mrc",
            "limit": None,
            "split": 10,
            "batch": 25,
            "no_val_drop": True,
            "affinity": os.path.join(self.testdata_mrc, "affinity_fsc_10.csv"),
            "classes": os.path.join(self.testdata_mrc, "classes.csv"),
            "collect_meta": True,
            "epochs": 5,
            "channels": 3,
            "depth": 4,
            "latent_dims": 8,
            "pose_dims": 3,
            "learning": 0.03,
            "beta_load": None,
            "beta_min": 0,
            "beta_max": 1,
            "beta": 1,
            "beta_cycle": "flat",
            "beta_ratio": None,
            "cyc_method_beta": "flat",
            "gamma_load": None,
            "gamma_min": 0,
            "gamma_max": 1,
            "gamma_cycle": None,
            "gamma_ratio": None,
            "gamma": 1,
            "cyc_method_gamma": "flat",
            "loss_fn": "MSE",
            "use_gpu": False,
            "restart": False,
            "state": None,
            "gpu": None,
            "meta": None,
            "gaussian_blur": True,
            "normalise": True,
            "shift_min": True,
            "tensorboard": True,
            "classifier": "NN",
            "new_out": False,
            "dynamic": True,
            "opt_method": "adam",
        }

        config.FREQ_ACC = 5
        config.FREQ_REC = 5
        config.FREQ_EMB = 5
        config.FREQ_INT = 5
        config.FREQ_DIS = 5
        config.FREQ_POS = 5
        config.FREQ_EVAL = 5
        config.FREQ_STA = 5
        config.FREQ_CON = 5
        config.FREQ_SIM = 5

        config.VIS_CYC = True
        config.VIS_LOS = True
        config.VIS_ACC = True
        config.VIS_REC = True
        config.VIS_EMB = True
        config.VIS_INT = True
        config.VIS_DIS = True
        config.VIS_POS = True
        config.VIS_HIS = True
        config.VIS_CON = True
        config.VIS_AFF = True
        config.VIS_SIM = True

    def test_model_a_mrc(self):
        self.data["model"] = "a"
        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
            n_plots_eval,
            n_latent_eval,
            n_states_eval,
        ) = helper_train_eval(self.data)

        self.assertEqual(n_dir_train, 4)
        self.assertGreaterEqual(n_plots_train, 32)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)

        self.assertGreaterEqual(n_plots_eval, 50)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)

    def test_model_b_mrc(self):
        self.data["model"] = "b"
        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
            n_plots_eval,
            n_latent_eval,
            n_states_eval,
        ) = helper_train_eval(self.data)

        self.assertEqual(n_dir_train, 4)
        self.assertGreaterEqual(n_plots_train, 32)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)
        self.assertGreaterEqual(n_plots_eval, 50)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)

    def test_model_a_npy(self):
        self.data["model"] = "a"
        self.data["datatype"] = "npy"
        self.data["datapath"] = self.testdata_npy
        self.data["affinity"] = os.path.join(
            self.testdata_npy, "affinity_an.csv"
        )
        self.data["classes"] = os.path.join(self.testdata_npy, "classes.csv")
        (
            n_dir_train,
            n_plots_train,
            n_latent_train,
            n_states_train,
            n_plots_eval,
            n_latent_eval,
            n_states_eval,
        ) = helper_train_eval(self.data)

        self.assertEqual(n_dir_train, 4)
        self.assertGreaterEqual(n_plots_train, 30)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)
        self.assertGreaterEqual(n_plots_eval, 47)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)


def helper_train_eval(data):

    temp_dir = tempfile.TemporaryDirectory()
    os.chdir(temp_dir.name)

    # run training
    data["eval"] = False
    run_pipeline(data)

    n_dir_train = len(next(os.walk(temp_dir.name))[1])
    n_plots_train = len(os.listdir(os.path.join(temp_dir.name, "plots")))
    n_latent_train = len(os.listdir(os.path.join(temp_dir.name, "latents")))
    n_states_train = len(os.listdir(os.path.join(temp_dir.name, "states")))

    # run evaluation
    data["eval"] = True
    data["datapath"] = os.path.join(data["datapath"], "test")
    run_pipeline(data)

    n_plots_eval = len(os.listdir(os.path.join(temp_dir.name, "plots")))
    n_latent_eval = len(os.listdir(os.path.join(temp_dir.name, "latents")))
    n_states_eval = len(os.listdir(os.path.join(temp_dir.name, "states")))

    shutil.rmtree(temp_dir.name)

    return (
        n_dir_train,
        n_plots_train,
        n_latent_train,
        n_states_train,
        n_plots_eval,
        n_latent_eval,
        n_states_eval,
    )


if __name__ == "__main__":
    unittest.main()
