from .. import psodataclass as pdc, codec
import unittest
import logging
import numpy as np
import os


_JSON_PATH = os.path.join(".", "pso", "test", "json", "")
_JSON_TEMP_PATH = os.path.join(_JSON_PATH, "temp", "")
print(_JSON_TEMP_PATH)

class DataTester(unittest.TestCase):
    """Base class for the other testing classes"""
    logging.basicConfig(level=logging.INFO)
    seed_int = 0

    def reset_seed(self) -> None:
        np.random.seed(self.seed_int)

    def setUp(self) -> None:
        self.reset_seed()
        return super().setUp()
    
class PSOHyperparametersTester(DataTester):
    def test_serialize(self):

        json_name = "PSOHyperparameterstest.json"
        file_path = f"{_JSON_TEMP_PATH}{json_name}"

        if os.path.exists(file_path):
            logging.info(f"Removing {json_name}")
            os.remove(file_path)

        expected_hparams = pdc.PSOHyperparameters(
            num_part = 1,
            num_dim = 2,
            alpha = 0.3,
            w = 0.4,
            c1 = 0.9,
            c2 = 0.8,
            tolerance = 0.3,
            mv_iteration = 78,
            max_iterations = 100
        )

        upper_bound = np.ones(5)
        lower_bound = np.ones(5)*-1

        expected_domaindata = pdc.DomainData(
            upper_bound=upper_bound,
            lower_bound=lower_bound
        )

        s = codec.dataclass_to_json(expected_hparams)
        print(s)
        z = codec.json_to_dataclass(s, pdc.PSOHyperparameters)
        print(z)

        s = codec.dataclass_to_json(expected_domaindata)
        print(s)
        z = codec.json_to_dataclass(s, pdc.DomainData)
        print(z)

        """
        expected_hparams.to_json_file(file_path)

        # Load the json file that we made
        test_hparams = pdc.PSOHyperparameters.from_json_file(file_path)

        # Make sure that the list of keys is the same first
        self.assertEqual(list(expected_hparams.__dict__.keys()), list(test_hparams.__dict__.keys()))

        # Test every key value in pair in the test and expected to make sure they are the same, 
        # and no loss has occurred during Serialization
        test_dict = test_hparams.__dict__
        expec_dict = expected_hparams.__dict__
        for key in test_dict:
            self.assertAlmostEqual(test_dict[key], expec_dict[key])
        """
        





if __name__ == "__main__":
    unittest.main()