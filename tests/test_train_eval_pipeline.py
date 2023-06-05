import os
import shutil
import tempfile
import unittest

from avae import config
from avae.evaluate import evaluate
from avae.train import train
from tests import testdata


class TrainEvalTest(unittest.TestCase):
    def setUp(self) -> None:
        """Test instantiation of the pipeline."""

        self.testdata = os.path.dirname(testdata.__file__)

        self.data = {
            "datapath": self.testdata,
            "lim": 100,
            "splt": 10,
            "batch_s": 25,
            "no_val_drop": True,
            "affinity": os.path.join(self.testdata, "affinity_fsc_10.csv"),
            "classes": os.path.join(self.testdata, "classes.csv"),
            "collect_meta": True,
            "epochs": 5,
            "channels": 3,
            "depth": 4,
            "lat_dims": 8,
            "pose_dims": 3,
            "learning": 0.03,
            "beta_min": 0,
            "beta_max": 1,
            "beta_cycle": "flat",
            "beta_ratio": None,
            "cyc_method_beta": "flat",
            "gamma_min": 0,
            "gamma_max": 1,
            "gamma_cycle": None,
            "gamma_ratio": None,
            "cyc_method_gamma": "flat",
            "recon_fn": "MSE",
            "use_gpu": False,
        }

        config.FREQ_ACC = 5
        config.FREQ_REC = 5
        config.FREQ_EMB = 5
        config.FREQ_INT = 5
        config.FREQ_DIS = 5
        config.FREQ_POS = 5
        config.FREQ_EVAL = 5
        config.FREQ_STA = 5

        config.VIS_CYC = True
        config.VIS_LOS = True
        config.VIS_ACC = True
        config.VIS_REC = True
        config.VIS_EMB = True
        config.VIS_INT = True
        config.VIS_DIS = True
        config.VIS_POS = True
        config.VIS_HIS = True

    def test_train_eval_a(self):

        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)

        train(
            datapath=self.data["datapath"],
            lim=self.data["lim"],
            splt=self.data["splt"],
            batch_s=self.data["batch_s"],
            no_val_drop=self.data["no_val_drop"],
            affinity=self.data["affinity"],
            classes=self.data["classes"],
            collect_meta=self.data["collect_meta"],
            epochs=self.data["epochs"],
            channels=self.data["channels"],
            depth=self.data["depth"],
            lat_dims=self.data["lat_dims"],
            pose_dims=self.data["pose_dims"],
            learning=self.data["learning"],
            beta_min=self.data["beta_min"],
            beta_max=self.data["beta_max"],
            beta_cycle=self.data["beta_cycle"],
            beta_ratio=self.data["beta_ratio"],
            cyc_method_beta=self.data["cyc_method_beta"],
            gamma_min=self.data["gamma_min"],
            gamma_max=self.data["gamma_max"],
            gamma_cycle=self.data["gamma_cycle"],
            gamma_ratio=self.data["gamma_ratio"],
            cyc_method_gamma=self.data["cyc_method_gamma"],
            recon_fn=self.data["recon_fn"],
            use_gpu=self.data["recon_fn"],
            model="b",
        )

        n_dir_train = len(next(os.walk(temp_dir.name))[1])
        n_plots_train = len(os.listdir(os.path.join(temp_dir.name, "plots")))
        n_latent_train = len(
            os.listdir(os.path.join(temp_dir.name, "latents"))
        )
        n_states_train = len(os.listdir(os.path.join(temp_dir.name, "states")))

        self.assertEqual(n_dir_train, 3)
        self.assertEqual(n_plots_train, 19)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)

        evaluate(
            datapath=os.path.join(self.testdata, "test"),
            lim=self.data["lim"],
            splt=self.data["splt"],
            batch_s=self.data["batch_s"],
            collect_meta=True,
            use_gpu=self.data["use_gpu"],
        )

        n_plots_eval = len(os.listdir(os.path.join(temp_dir.name, "plots")))
        n_latent_eval = len(os.listdir(os.path.join(temp_dir.name, "latents")))
        n_states_eval = len(os.listdir(os.path.join(temp_dir.name, "states")))

        self.assertEqual(n_plots_eval, 28)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 2)

        shutil.rmtree(temp_dir.name)

    def test_train_eval_b(self):

        temp_dir = tempfile.TemporaryDirectory()
        os.chdir(temp_dir.name)

        train(
            datapath=self.data["datapath"],
            lim=self.data["lim"],
            splt=self.data["splt"],
            batch_s=self.data["batch_s"],
            no_val_drop=self.data["no_val_drop"],
            affinity=self.data["affinity"],
            classes=self.data["classes"],
            collect_meta=self.data["collect_meta"],
            epochs=self.data["epochs"],
            channels=self.data["channels"],
            depth=self.data["depth"],
            lat_dims=self.data["lat_dims"],
            pose_dims=self.data["pose_dims"],
            learning=self.data["learning"],
            beta_min=self.data["beta_min"],
            beta_max=self.data["beta_max"],
            beta_cycle=self.data["beta_cycle"],
            beta_ratio=self.data["beta_ratio"],
            cyc_method_beta=self.data["cyc_method_beta"],
            gamma_min=self.data["gamma_min"],
            gamma_max=self.data["gamma_max"],
            gamma_cycle=self.data["gamma_cycle"],
            gamma_ratio=self.data["gamma_ratio"],
            cyc_method_gamma=self.data["cyc_method_gamma"],
            recon_fn=self.data["recon_fn"],
            use_gpu=self.data["recon_fn"],
            model="b",
        )

        n_dir_train = len(next(os.walk(temp_dir.name))[1])
        n_plots_train = len(os.listdir(os.path.join(temp_dir.name, "plots")))
        n_latent_train = len(
            os.listdir(os.path.join(temp_dir.name, "latents"))
        )
        n_states_train = len(os.listdir(os.path.join(temp_dir.name, "states")))

        self.assertEqual(n_dir_train, 3)
        self.assertEqual(n_plots_train, 19)
        self.assertEqual(n_latent_train, 2)
        self.assertEqual(n_states_train, 2)

        evaluate(
            datapath=os.path.join(self.testdata, "test"),
            lim=self.data["lim"],
            splt=self.data["splt"],
            batch_s=self.data["batch_s"],
            collect_meta=True,
            use_gpu=self.data["use_gpu"],
        )

        n_plots_eval = len(os.listdir(os.path.join(temp_dir.name, "plots")))
        n_latent_eval = len(os.listdir(os.path.join(temp_dir.name, "latents")))
        n_states_eval = len(os.listdir(os.path.join(temp_dir.name, "states")))

        self.assertEqual(n_plots_eval, 28)
        self.assertEqual(n_latent_eval, 4)
        self.assertEqual(n_states_eval, 2)

        shutil.rmtree(temp_dir.name)


if __name__ == "__main__":
    unittest.main()