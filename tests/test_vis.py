import os
import random
import unittest

import torch

from avae import config, settings
from tests import testdata_mrc
from tests.test_train_eval_pipeline import helper_train_eval

torch.random.manual_seed(10)
random.seed(10)


class VisPipelineTest(unittest.TestCase):
    """Test pipeline with isolated visualisations."""

    def setUp(self) -> None:
        """Setup data and output directories."""
        self.test_data = os.path.dirname(testdata_mrc.__file__)

        self.data_params = {  # only specify if differs from default
            # data
            "datapath": self.test_data,
            "affinity": os.path.join(self.test_data, "affinity_fsc_10.csv"),
            "classes": os.path.join(self.test_data, "classes.csv"),
            "split": 10,
            # preprocess
            "rescale": 32,
            "gaussian_blur": True,
            "normalise": True,
            "shift_min": True,
            # model
            "epochs": 1,
            "batch": 25,
            "model": "u",
            "channels": 3,
            "depth": 4,
            "latent_dims": 8,
            "pose_dims": 3,
            "learning": 0.03,
            "beta": 1,
            "gamma": 1,
            # vis
            "vis_all": False,
            "freq_all": 1,
            "vis_format": "png",
        }

        self.data = config.load_config_params(local_vars=self.data_params)
        config.setup_visualisation_config(self.data)

    def test_accuracy(self):
        settings.VIS_ACC = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 7)
        # confusion val and train + val and train norm, f1, f1 val and f1 train

    def test_loss(self):
        self.data["epochs"] = 2
        settings.VIS_LOS = True

        __, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 3)  # loss, total loss, train loss

    def test_recon(self):
        settings.VIS_REC = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 7)
        # recon in and out for train and val + 3D + reconstructions dir

    def test_similarity(self):
        settings.VIS_SIM = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 2)  # recon and val

    def test_embedding(self):
        settings.VIS_EMB = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 4)  # tsne and umap for lat+pose

    def test_dyn_embedding(self):
        settings.VIS_EMB = True
        settings.VIS_DYN = True

        _, n_plots, n_latent, _ = helper_train_eval(
            self.data, eval=False, nostate=True
        )

        self.assertEqual(n_plots, 4)
        self.assertEqual(n_latent, 2)  # tsne and umap

    def test_latent_disentanglement(self):
        settings.VIS_DIS = True
        lat_pose = [(0, 0), (3, 0), (0, 3), (3, 0)]

        for l, p in lat_pose:
            self.data["latent_dim"] = l
            self.data["pose_dim"] = p

            _, n_plots, _, _ = helper_train_eval(
                self.data, eval=False, nolat=True, nostate=True
            )

            self.assertEqual(n_plots, 1)  # in future 3 + per class

    def test_pose_disentanglement(self):
        settings.VIS_POS = True
        self.data["pose_dims"] = 0

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 0)

        class_out = [(None, 1), ("1b23", 2), ("1b23,1dkg", 3)]

        for c, out in class_out:
            self.data["pose_dims"] = 3
            settings.VIS_POSE_CLASS = c

            _, n_plots, _, _ = helper_train_eval(
                self.data, eval=False, nolat=True, nostate=True
            )

            self.assertEqual(n_plots, out)  # in future 3 + per class

    def test_interpolation(self):
        settings.VIS_INT = True
        lat_pose = [(0, 0), (3, 0), (0, 3), (3, 0)]

        for l, p in lat_pose:
            self.data["latent_dim"] = l
            self.data["pose_dim"] = p

            _, n_plots, _, _ = helper_train_eval(
                self.data, eval=False, nolat=True, nostate=True
            )

            self.assertEqual(n_plots, 1)  # in future 3

    def test_affinity(self):
        settings.VIS_AFF = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 1)

    def test_distribution(self):
        settings.VIS_HIS = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 2)  # train and val

    def test_cyc_variables(self):
        settings.VIS_CYC = True

        _, n_plots, _, _ = helper_train_eval(
            self.data, eval=False, nolat=True, nostate=True
        )

        self.assertEqual(n_plots, 2)  # beta and gamma
