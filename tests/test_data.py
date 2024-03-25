import os
import shutil
import tempfile
import unittest

import lightning as lt
import numpy as np

from avae.data import load_data
from tests import testdata_mrc


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        """Setup data and output directories."""
        self._orig_dir = os.getcwd()
        self.test_data = os.path.dirname(testdata_mrc.__file__)
        self.test_dir = tempfile.mkdtemp(prefix="avae_")
        self.fabric = lt.Fabric()
        self.fabric.launch()

        # Change to test directory
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_eval_data(self):
        """Test loading evaluation data."""

        sh = 32

        shutil.copytree(
            os.path.join(self.test_data, "test"),
            os.path.join(self.test_dir, "eval"),
        )

        out, data_dim = load_data(
            "./eval",
            datatype="mrc",
            lim=None,
            batch_s=32,
            eval=True,
            gaussian_blur=True,
            normalise=True,
            shift_min=True,
            rescale=sh,
            fabric=self.fabric,
        )
        print(os.getcwd())

        # test load_data
        assert len(out) == 1
        eval_data = out
        # assert isinstance(eval_data, lt.fast.DataLoader)

        # test ProteinDataset
        eval_batch = list(eval_data)[0]
        xs, ys, aff, meta = eval_batch
        assert len(xs) == len(ys) == len(aff)
        assert (
            np.all(aff.numpy()) == 0
        )  # this is expected only for eval without affinity

        assert xs[0].shape[-1] == sh

    def test_load_train_data(self):
        """Test loading training data."""
        shutil.copytree(self.test_data, os.path.join(self.test_dir, "train"))

        sh = 32

        out = load_data(
            "./train",
            datatype="mrc",
            lim=None,
            splt=30,
            batch_s=16,
            no_val_drop=True,
            eval=False,
            affinity_path="./train/affinity_fsc_10.csv",
            gaussian_blur=True,
            normalise=True,
            shift_min=True,
            rescale=sh,
            fabric=self.fabric,
        )

        # test load_data
        assert len(out) == 5
        train_data, val_data, test_data, lookup, data_dim = out
        assert len(train_data) >= len(val_data)
        # assert isinstance(train_data, lt.fabric.DataLoader)

        # test ProtenDataset
        train_batch = list(train_data)[0]
        xs, ys, aff, meta = train_batch
        assert len(xs) == len(ys) == len(aff)
        assert len(np.unique(aff.numpy())) == 4
        assert xs[0].shape[-1] == sh

        # test affinity matrix
        assert isinstance(lookup, np.ndarray)
        assert len(lookup.shape) == 2
        assert lookup.shape[0] == lookup.shape[1]
        assert lookup[0][0] == 1

        data_0 = list(out[0])[0][0][0][0][0].detach().numpy()
        data_1 = list(out[0])[0][0][0][0][0].detach().numpy()

        # test reproducibility
        out_2, _, _, _, _ = load_data(
            "./train",
            datatype="mrc",
            lim=None,
            splt=30,
            batch_s=16,
            no_val_drop=True,
            eval=False,
            affinity_path="./train/affinity_fsc_10.csv",
            gaussian_blur=True,
            normalise=True,
            shift_min=True,
            rescale=sh,
            fabric=self.fabric,
        )
        data_0_1 = list(out_2)[0][0][0][0][0].detach().numpy()
        data_1_1 = list(out_2)[0][0][0][0][0].detach().numpy()

        assert data_0.all() == data_0_1.all()
        assert data_1.all() == data_1_1.all()
