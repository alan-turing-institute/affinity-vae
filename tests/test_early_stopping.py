import random
import unittest

import numpy as np

from avae.utils_learning import EarlyStopping

# fixing random seeds so we dont get fail on mrc tests
random.seed(10)


class EarlyStopTest(unittest.TestCase):
    def setUp(self) -> None:

        self.stopper_total = EarlyStopping(
            loss_type="total_loss",
            patience=10,
            max_delta=0.01,
            max_divergence=0.2,
            min_epochs=20,
        )
        self.stopper_kld = EarlyStopping(
            loss_type="kldiv_loss",
            patience=10,
            max_delta=0.01,
            max_divergence=0.2,
            min_epochs=20,
        )
        self.stopper_affinity = EarlyStopping(
            loss_type="affinity_loss",
            patience=10,
            max_delta=0.1,
            max_divergence=0.2,
            min_epochs=20,
        )
        self.stopper_reco = EarlyStopping(
            loss_type="reco_loss",
            patience=10,
            max_delta=0.01,
            max_divergence=0.2,
            min_epochs=20,
        )

        recon_loss_train = np.linspace(10, 0.001, 100)
        kldivergence_train = np.linspace(0.1, 0.001, 100)
        affin_loss_train = np.linspace(0.1, 0.001, 100)
        # total loss
        total_loss_train = (
            recon_loss_train + kldivergence_train + affin_loss_train
        )

        self.loss_train = [
            total_loss_train,
            recon_loss_train,
            kldivergence_train,
            affin_loss_train,
        ]

        self.recon_loss_val = np.linspace(10, 0.001, 100)
        self.kldivergence_val = np.linspace(1, 0.1, 100)

        self.affin_loss_val = np.linspace(0.1, 0.001, 100)

        # total loss
        self.total_loss_val = (
            self.recon_loss_val + self.kldivergence_val + self.affin_loss_val
        )

        self.loss_val = [
            self.total_loss_val,
            self.recon_loss_val,
            self.kldivergence_val,
            self.affin_loss_val,
        ]

    def test_early_stopping_decreasing_loss(self):

        stop_total = self.stopper_total.early_stop(
            self.loss_val, self.loss_train
        )
        stop_recon = self.stopper_reco.early_stop(
            self.loss_val, self.loss_train
        )
        stop_kldiv = self.stopper_kld.early_stop(
            self.loss_val, self.loss_train
        )
        stop_affinity = self.stopper_affinity.early_stop(
            self.loss_val, self.loss_train
        )

        assert stop_total is False
        assert stop_recon is False
        assert stop_kldiv is False
        assert stop_affinity is False

    def test_early_stopping_increasing_single_loss(self):

        kldivergence_val = np.linspace(0.001, 0.1, 100)

        affin_loss_val = np.linspace(0.1, 0.01, 100)

        # total loss
        total_loss_val = (
            self.recon_loss_val + kldivergence_val + affin_loss_val
        )

        loss_val = [
            total_loss_val,
            self.recon_loss_val,
            kldivergence_val,
            affin_loss_val,
        ]

        stop_total = self.stopper_total.early_stop(loss_val, self.loss_train)
        stop_kldiv = self.stopper_kld.early_stop(loss_val, self.loss_train)
        stop_affinity = self.stopper_affinity.early_stop(
            loss_val, self.loss_train
        )

        assert stop_total is False
        assert stop_kldiv is True
        assert stop_affinity is True

    def test_early_stopping_increasing_total_loss(self):

        kldivergence_val = np.linspace(0.1, 10, 100)

        # total loss
        total_loss_val = (
            self.recon_loss_val + kldivergence_val + self.affin_loss_val
        )

        loss_val = [
            total_loss_val,
            self.recon_loss_val,
            kldivergence_val,
            self.affin_loss_val,
        ]

        stop_total = self.stopper_total.early_stop(loss_val, self.loss_train)
        stop_kldiv = self.stopper_kld.early_stop(loss_val, self.loss_train)

        assert stop_total is True
        assert stop_kldiv is True

    def test_early_stopping_fluctuating_loss(self):

        recon_loss_val = np.random.uniform(1, 1.01, 100)

        # total loss
        total_loss_val = (
            recon_loss_val + self.kldivergence_val + self.affin_loss_val
        )

        loss_val = [
            total_loss_val,
            recon_loss_val,
            self.kldivergence_val,
            self.affin_loss_val,
        ]

        stop_total = self.stopper_total.early_stop(loss_val, self.loss_train)
        stop_recon = self.stopper_reco.early_stop(loss_val, self.loss_train)

        assert stop_total is True
        assert stop_recon is True

    def test_early_stopping_decreasing_loss_within_tolerance(self):

        kldivergence_val = np.random.uniform(1, 1.1, 100)

        affin_loss_val = np.linspace(0.1, 0.0009, 100)

        # total loss
        total_loss_val = (
            self.recon_loss_val + kldivergence_val + affin_loss_val
        )

        loss_val = [
            total_loss_val,
            self.recon_loss_val,
            kldivergence_val,
            affin_loss_val,
        ]

        stop_recon = self.stopper_reco.early_stop(loss_val, self.loss_train)
        stop_affinity = self.stopper_affinity.early_stop(
            loss_val, self.loss_train
        )

        assert stop_affinity is False
        assert stop_recon is False
