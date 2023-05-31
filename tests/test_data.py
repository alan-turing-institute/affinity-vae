import os
import shutil
import tempfile
import unittest

import numpy as np
from torch.utils.data import DataLoader

from avae.data import load_data
from tests import testdata


class DataTest(unittest.TestCase):
    def setUp(self) -> None:
        """Setup data and output directories."""
        self.test_data = os.path.dirname(testdata.__file__)
        self.test_dir = tempfile.mkdtemp(prefix="avae_")
        print(self.test_data, self.test_dir)

        # Change to test directory
        self._orig_dir = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self._orig_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_eval_data(self):
        """Test loading evaluation data."""

        shutil.copytree(
            os.path.join(self.test_data, "test"),
            os.path.join(self.test_dir, "eval"),
        )

        out = load_data(
            "./eval", lim=None, batch_s=32, collect_meta=False, eval=True
        )
        print(os.getcwd())

        # test load_data
        assert len(out) == 1
        eval_data = out
        assert isinstance(eval_data, DataLoader)

        # test ProteinDataset
        eval_batch = list(eval_data)[0]
        xs, ys, aff = eval_batch
        assert len(xs) == len(ys) == len(aff)
        assert (
            np.all(aff.numpy()) == 0
        )  # this is expected only for eval without affinity

    def test_load_train_data(self):
        """Test loading training data."""
        shutil.copytree(self.test_data, os.path.join(self.test_dir, "train"))

        out = load_data(
            "./train",
            lim=None,
            splt=30,
            batch_s=16,
            no_val_drop=True,
            collect_meta=False,
            eval=False,
            affinity="./train/affinity_fsc_10.csv",
        )

        # test load_data
        assert len(out) == 4
        train_data, val_data, test_data, lookup = out
        assert len(train_data) >= len(val_data)
        assert isinstance(train_data, DataLoader)

        # test ProtenDataset
        train_batch = list(train_data)[0]
        xs, ys, aff = train_batch
        assert len(xs) == len(ys) == len(aff)
        assert len(np.unique(aff.numpy())) == 4

        # test affinity matrix
        assert isinstance(lookup, np.ndarray)
        assert len(lookup.shape) == 2
        assert lookup.shape[0] == lookup.shape[1]
        assert lookup[0][0] == 1

    # write more tests for ProteinDataset
