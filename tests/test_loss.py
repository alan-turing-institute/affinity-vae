import unittest

import numpy as np
import pandas as pd
import torch
import os
from avae.loss import AVAELoss
from tests import testdata


class LossTest(unittest.TestCase):
    def setUp(self) -> None:
        """Test instantiation of the loss."""

        self.test_data = os.path.dirname(testdata.__file__)
        os.chdir(self.test_dir)


        self.affinity = pd.read_csv("./testdata/affinity_fsc_10.csv").to_numpy(
            dtype=np.float32
        )
        self.loss = AVAELoss(
            torch.device("cpu"),
            [1],
            [1],
            lookup_aff=self.affinity,
            recon_fn="MSE",
        )

    def test_loss_instatiation(self):
        """Test instantiation of the loss."""

        assert isinstance(self.loss, AVAELoss)
