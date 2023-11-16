import os
import random
import shutil
import tempfile
import unittest

import torch

from avae import config, settings
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

        self.data_params = {
            "datapath": self.testdata_mrc,
            "datatype": "mrc",
            "split": 10,
            "batch": 25,
            "no_val_drop": True,
            "affinity": os.path.join(self.testdata_mrc, "affinity_fsc_10.csv"),
            "classes": os.path.join(self.testdata_mrc, "classes.csv"),
            "dynamic": True,
            "epochs": 5,
            "channels": 3,
            "depth": 4,
            "latent_dims": 8,
            "pose_dims": 3,
            "learning": 0.03,
            "beta_min": 0,
            "beta": 1,
            "beta": 1,
            "beta_cycle": 1,
            "cyc_method_beta": "flat",
            "gamma_min": 0,
            "gamma": 1,
            "gamma": 1,
            "cyc_method_gamma": "flat",
            "loss_fn": "MSE",
            "gaussian_blur": True,
            "normalise": True,
            "shift_min": True,
            "rescale": 32,
            "tensorboard": True,
            "classifier": "NN",
            "opt_method": "adam",
            "freq_all": 5,
            "vis_all": True,
        }

        self.data = config.load_config_params(local_vars=self.data_params)

        config.setup_visualisation_config(self.data)

    def test_model_a_mrc(self):
        self.data["model"] = "a"
        settings.VIS_POSE_CLASS = "1b23,1dkg"

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
        self.assertEqual(n_plots_train, 34)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)

        self.assertEqual(n_plots_eval, 52)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)

    def test_model_b_mrc(self):
        self.data["model"] = "b"
        settings.VIS_POSE_CLASS = "1b23,1dkg"

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
        self.assertEqual(n_plots_train, 34)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)
        self.assertEqual(n_plots_eval, 52)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)

    def test_model_a_npy(self):
        self.data["model"] = "a"
        self.data["datatype"] = "npy"
        self.data["datapath"] = self.testdata_npy
        settings.VIS_POSE_CLASS = "2,5"

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
        self.assertEqual(n_plots_train, 32)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)
        self.assertEqual(n_plots_eval, 49)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 3)

    def test_model_b_npy(self):
        self.data["model"] = "b"
        self.data["datatype"] = "npy"
        self.data["datapath"] = self.testdata_npy
        settings.VIS_POSE_CLASS = "2,5"

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
        self.assertEqual(n_plots_train, 32)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)
        self.assertEqual(n_plots_eval, 49)
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
