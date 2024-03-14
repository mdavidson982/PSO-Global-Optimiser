from .. import psodataclass as dc, codec
import unittest
import logging
import numpy as np
import os

_JSON_PATH = os.path.join(".", "pso", "test", "json", "")
_JSON_TEMP_PATH = os.path.join(_JSON_PATH, "temp", "")

class DataTester(unittest.TestCase):
    """Base class for the other testing classes"""
    logging.basicConfig(level=logging.INFO)
    seed_int = 0

    def reset_seed(self) -> None:
        np.random.seed(self.seed_int)

    def setUp(self) -> None:
        self.reset_seed()
        return super().setUp()
    
    @classmethod
    def setUpClass(cls) -> None:
        # Clean up any old files in the temp directory
        logging.info("Removing files in temp directory")
        for item in os.listdir(_JSON_TEMP_PATH):
            os.remove(f"{_JSON_TEMP_PATH}{item}")
    
class PSOCodecTester(DataTester):
    """Checks that the codec json serialization/deserialization works as intended"""
    def check_equals(self, expected, real):
        """Helper function to determine if two classes """
        # Make sure that the list of keys is the same first
        self.assertEqual(type(expected), type(real))
        self.assertEqual(list(expected.__dict__.keys()), list(real.__dict__.keys()))

        # Test every key value in pair in the test and expected to make sure they are the same, 
        # and no loss has occurred during Serialization
        test_dict = real.__dict__
        expec_dict = expected.__dict__
        for key, value in test_dict.items():
            if type(value) == np.ndarray:
                np.testing.assert_array_almost_equal(test_dict[key], expec_dict[key], 8)
            else:
                self.assertAlmostEqual(test_dict[key], expec_dict[key], 8)

    def test_hparams_codec(self):
        """Test that the codec can serialize and deserialize objects successfully"""
        json_name = "PSOCodecHparamsTest.json"
        file_path = f"{_JSON_TEMP_PATH}{json_name}"

        expected_hparams = dc.PSOHyperparameters(
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

        # Check regular serialization works
        hparams_jsonized = codec.dataclass_to_json(expected_hparams)
        unjsonized_hparams = codec.json_to_dataclass(hparams_jsonized, dc.PSOHyperparameters)
        self.check_equals(expected_hparams, unjsonized_hparams)

        # Check that the json can be written to and retrieved from file
        with open(file_path, "w+") as file:
            codec.dataclass_to_json_file(expected_hparams, file)

        with open(file_path, "r") as file:
            file_unjsonized_hparams = codec.json_file_to_dataclass(file, dc.PSOHyperparameters)
        self.check_equals(expected_hparams, file_unjsonized_hparams)


    def test_domain_codec(self):
        """
        Check that domain, which uses numpy arrays, is able to be fully serialized to and from a string, and from a file
        """
        json_name = "PSOCodecDomainTest.json"
        file_path = f"{_JSON_TEMP_PATH}{json_name}"

        # Check a more difficult case, like DomainData, which has np arrays (more difficult to jsonize)
        # NOTE the values used below should not be used as a basis for constructing mpso instances
        upper_bound = np.ones(shape=(5, 2))
        lower_bound = np.ones(5)*-1
        expected_domaindata = dc.DomainData(
            upper_bound=upper_bound,
            lower_bound=lower_bound
        )

        # Check regular serialization works
        domaindata_jsonized = codec.dataclass_to_json(expected_domaindata)
        unjsonized_domaindata = codec.json_to_dataclass(domaindata_jsonized, dc.DomainData)
        self.check_equals(expected_domaindata, unjsonized_domaindata)

        # Check that the json can be written to and retrieved from file
        with open(file_path, "w+") as file:
            codec.dataclass_to_json_file(expected_domaindata, file)
        with open(file_path, "r") as file:
            file_unjsonized_domaindata = codec.json_file_to_dataclass(file, dc.DomainData)
        self.check_equals(expected_domaindata, file_unjsonized_domaindata)

    def test_read_in(self):
        file_names = [(file_name[0]+".json", file_name[1]) for file_name in [
            ("ccdparams", dc.CCDHyperparameters),
            ("domain", dc.DomainData),
            ("hparams", dc.PSOHyperparameters),
            ("loggerconfig", dc.PSOLoggerConfig),
            ("mpsoconfig", dc.MPSORunnerConfigs),
            ("psoconfig", dc.PSOConfig),
        ]]

        # Try to read in all of the json files in the json folder,
        # and deserialize to their respective types
        for file_data in file_names:
            with open(f"{_JSON_PATH}{file_data[0]}", "r") as file:
                dataclass = codec.json_file_to_dataclass(file, file_data[1])
                logging.info(f"Constructed {file_data[1].__name__}")
                logging.info(f"Contents: {dataclass.__dict__}\n")
        
                

if __name__ == "__main__":
    unittest.main()