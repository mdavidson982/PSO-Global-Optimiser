import numpy as np
from . import parameters as p
from . import util
import logging
import unittest

class UtilTester(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)
    seed_int = 0

    def reset_seed(self) -> None:
        np.random.seed(self.seed_int)

    def setUp(self) -> None:
        self.reset_seed()
        return super().setUp()
    
    def test_scale(self):
        """Test scaling function"""
        test_lb = np.array((-1, -3), dtype = p.DTYPE)
        test_ub = np.array((7, 9), dtype=p.DTYPE)

        #test scaling for 2d array
        array_2d = np.array((
            (0.25, 0.50, 0.75),
            (0.50, 0.25, 0.75)
        ), dtype=np.float64)
        test_scale_2d = util.scale(test_lb, test_ub, array_2d)
        expected_scale_2d = np.array((
            (1, 3, 5),
            (3, 0, 6)
        ), dtype=np.float64)
        np.testing.assert_array_almost_equal(test_scale_2d, expected_scale_2d, 8)

        #test scaling for 1d array
        array_1d = np.array((0, 0.125), dtype=np.float64)
        test_scale_1d = util.scale(test_lb, test_ub, array_1d)
        expected_scale_1d = np.array((-1, -1.5), dtype=np.float64)
        np.testing.assert_array_almost_equal(test_scale_1d, expected_scale_1d, 8)


    def test_descale(self):
        test_lb = np.array((-3, -2), dtype = p.DTYPE)
        test_ub = np.array((5, 18), dtype=p.DTYPE)

        #test descaling for 2d array
        array_2d = np.array((
            (-2, 3, 4),
            (5, 8, -1)
        ), dtype=np.float64)
        test_descale_2d = util.descale(test_lb, test_ub, array_2d)
        expected_descale_2d = np.array((
            (0.125, 0.75, 0.875),
            (0.35, 0.5, 0.05)
        ), dtype=np.float64)
        np.testing.assert_array_almost_equal(test_descale_2d, expected_descale_2d, 8)

        #test descaling for 1d array
        array_1d = np.array((4, 13), dtype=np.float64)
        test_descale_1d = util.descale(test_lb, test_ub, array_1d)
        expected_descale_1d = np.array((0.875, 0.75), dtype=np.float64)
        np.testing.assert_array_almost_equal(test_descale_1d, expected_descale_1d, 8)

    def test_project(self):
        test_old_lb = np.array((-5, -4))
        test_old_ub = np.array((5, 4))

        test_new_lb = np.array((10, 8))
        test_new_ub = np.array((30, 16))

        array = np.array((
            (3, 2, -1),
            (-4, 0, 4)
        ), dtype=np.float64)

        test_project = util.project(
            old_lb = test_old_lb, 
            old_ub = test_old_ub, 
            new_lb = test_new_lb, 
            new_ub = test_new_ub, 
            array = array
        )

        expected_project = np.array((
            (26, 24, 18),
            (8, 12, 16)
        ), dtype=np.float64)

        np.testing.assert_array_almost_equal(test_project, expected_project, 8)