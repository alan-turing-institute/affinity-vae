import glob
import os
import tempfile
import unittest

import avae.settings as settings
import configs
from avae.config import AffinityConfig, load_config_params, write_config_file
from tests import testdata_mrc, testdata_npy


class ConfigTest(unittest.TestCase):
    def __init__(self, methodName: str = ...):
        super().__init__(methodName)

    def setUp(self) -> None:
        """Setup data and output directories."""

        self.config = os.path.join(
            os.path.dirname(configs.__file__), "avae-test-config.yml"
        )
        self.data_local = {
            "datapath": os.path.dirname(testdata_mrc.__file__),
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "split": 5,
            "epochs": 100,
            "vis_sim": True,
        }

        self.datapath_local_missing = {
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "split": 5,
            "epochs": 100,
        }

        self.data_local_fail_wrong_label = {
            "datapath": os.path.dirname(testdata_mrc.__file__),
            "affinity": os.path.join(
                os.path.dirname(testdata_mrc.__file__), "affinity_fsc_10.csv"
            ),
            "classes": os.path.join(
                os.path.dirname(testdata_npy.__file__), "classes.csv"
            ),
            "collect_meta": True,
            "batch": "25",
        }
        self.data_local_fail_wrong_type = {
            "batch": "25",
            "split": "test",
        }

        self.default_model = AffinityConfig()

    def test_validate_config(self):

        data = load_config_params(
            config_file=self.config, local_vars=self.data_local
        )

        self.assertEqual(
            len(data.items()), len(self.default_model.model_dump().items())
        )
        self.assertEqual(data["batch"], 128)
        self.assertEqual(data["affinity"], self.data_local['affinity'])
        self.assertEqual(data["new_out"], self.default_model.new_out)
        self.assertEqual(data["split"], self.data_local['split'])
        self.assertEqual(data["epochs"], self.data_local['epochs'])
        self.assertEqual(data["vis_sim"], self.data_local['vis_sim'])

        data_config_only = load_config_params(config_file=self.config)
        self.assertEqual(data_config_only["epochs"], 1000)

        data_local_data_only = load_config_params(local_vars=self.data_local)
        self.assertEqual(data_local_data_only["epochs"], 100)
        self.assertEqual(
            data_local_data_only["vis_all"], self.default_model.vis_all
        )

    def test_validate_config_fail(self):

        with self.assertRaises(ValueError):
            load_config_params(local_vars=self.data_local_fail_wrong_label)

        with self.assertRaises(TypeError):
            load_config_params(local_vars=self.data_local_fail_wrong_type)

        with self.assertRaises(ValueError):
            load_config_params(local_vars=self.datapath_local_missing)

        # wrong input for classifier
        self.data_local['classifier'] = 'LS'
        with self.assertRaises(TypeError):
            load_config_params(local_vars=self.data_local)

    def test_write_config_file(self):
        temp_dir = tempfile.TemporaryDirectory()

        data = load_config_params(
            config_file=self.config, local_vars=self.data_local
        )
        os.chdir(temp_dir.name)

        write_config_file(settings.date_time_run, data)

        files = glob.glob(
            temp_dir.name + "/configs/*" + settings.date_time_run + "*.yaml"
        )

        data_from_output = load_config_params(config_file=files[0])

        self.assertEqual(len(files), 1)

        data['datapath'] = data_from_output['datapath']
        data['affinity'] = data_from_output['affinity']
        data['classes'] = data_from_output['classes']

        self.assertEqual(data_from_output, data)

        temp_dir.cleanup()
