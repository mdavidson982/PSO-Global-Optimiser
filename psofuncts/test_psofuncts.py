import unittest
from . import initializer, update, ccd
import logging
import numpy as np
import testfuncts.testfuncts as tf
from utils import util

class PSOTester(unittest.TestCase):
    """Base class for the other testing classes"""
    logging.basicConfig(level=logging.INFO)
    seed_int = 0

    def reset_seed(self) -> None:
        np.random.seed(self.seed_int)

    def setUp(self) -> None:
        self.reset_seed()
        return super().setUp()

class PSOInitializser(PSOTester):
    """Tests the PSO initializer functions"""
    def test_x_initializer(self):
        """Test to make sure that the position matrix initializes correctly"""
        # Test to make sure initialization works
        num_dim = 2
        num_part = 2
        upper_bound = np.ones((num_dim))*5
        lower_bound = np.ones((num_dim))*-4

        pos_matrix_expected = np.array(((0.93932154, 2.4367043), (1.42487038, 0.90394865)))
        pos_matrix_test = initializer._x_initializer(num_dim, num_part, upper_bound, lower_bound)

        # Test to make sure that the position matrix matches what we get experimentally for the 0 seed
        np.testing.assert_array_almost_equal(pos_matrix_expected, pos_matrix_test, decimal=8)

    def test_v_initializer(self):
        """Test to make sure that the velocity and vmax matrices initializes correctly"""
        num_dim = 2
        num_part = 2
        upper_bound = np.ones((num_dim))*6
        lower_bound = np.ones((num_dim))*-7
        alpha = 0.3

        v_matrix_expected = np.array(((0.38074533, 1.67847706), (0.80155433, 0.35008883)))
        v_max_expected = np.array((3.9, 3.9))

        v_matrix_test, v_max_test = initializer._v_initializer(num_dim, num_part, upper_bound, lower_bound, alpha)
        # Test to make sure that the velocity and vmax matrices match what we get experimentally for the 0 seed
        np.testing.assert_array_almost_equal(v_matrix_expected, v_matrix_test, 8)
        np.testing.assert_array_almost_equal(v_max_expected, v_max_test, 8)

    def test_initializer(self):
        """Test to make sure initialization function happens correctly"""
        num_dim = 2
        num_part = 2
        upper_bound = np.ones((num_dim))*7
        lower_bound = np.ones((num_dim))*-8
        optimum = np.zeros((num_dim))

        sphere_func = tf.TestFuncts.generate_function("sphere", optimum=optimum, bias = 0)
        
        pos_matrix_test, vel_matrix_test, p_best_test, g_best_test, v_max_test = initializer.initializer(
            num_part = num_part,
            num_dim = num_dim,
            alpha = 0.2,
            upper_bound = upper_bound,
            lower_bound = lower_bound,
            function = sphere_func
        )

        # Experimental results when initializing with seed at 0
        pos_matrix_expected = np.array(((0.23220256, 2.7278405), (1.04145064, 0.17324774)))
        vel_matrix_expected = np.array(((-0.4580712 ,  0.87536468), (-0.37447673,  2.350638)))
        v_max_expected = np.array((3, 3))

        # Ensure that the p_best matrix is the same when not using vectorized operations
        p_best_val = []
        min_val = np.finfo(np.float64).max
        min_ind = 0
        for i in range(pos_matrix_expected.shape[1]):
            val = sphere_func(pos_matrix_expected[:, i])
            if val < min_val:
                min_ind = i
                min_val = val
            p_best_val.append(val)

        p_best_val = np.array(p_best_val)
        p_best_expected = np.vstack((pos_matrix_expected, p_best_val))

        # g_best column (particle) should be the minimum value in the last row of the p_best matrix
        g_best_expected = p_best_expected[:, min_ind]

        # Check to make sure all of the returned values match what we expect
        np.testing.assert_array_almost_equal(pos_matrix_test, pos_matrix_expected, 8)
        np.testing.assert_array_almost_equal(vel_matrix_test, vel_matrix_expected, 8)
        np.testing.assert_array_almost_equal(v_max_test, v_max_expected, 8)
        np.testing.assert_array_almost_equal(p_best_test, p_best_expected, 7)
        np.testing.assert_array_almost_equal(g_best_test, g_best_expected, 8)

class PSOUpdater(PSOTester):
    """Tests the updater functions for PSO"""
    def test_update_g_best(self):
        """Check to make sure that update_g_best works as intended"""
        test_p_best = np.array((
            (1, 2),
            (3, 4),
            (0, 1)
        ))

        # Best must be the first column, which has a value of 0
        g_best_expected = test_p_best[:, 0]
        g_best_test = update.update_g_best(test_p_best)
        np.testing.assert_array_almost_equal(g_best_test, g_best_expected, 8)

        # Test to make sure that the returned array does not interfere with the pbest
        # Will fail if g_best is treated as a pointer and not an independent array
        test_p_best[0, 0] += 1
        self.assertNotAlmostEqual(test_p_best[0, 0], g_best_test[0])

    def test_update_p_best(self):
        """Check to mmake sure that update_p_best works as intended"""
        optimum = np.zeros((2))
        sphere_func = tf.TestFuncts.generate_function("sphere", optimum=optimum, bias = 0)

        # Position matrix, which for sphere, should evaluate to 25 and 100 for each respective column
        pos_matrix = np.array((
            (3, 6),
            (4, 8)
            #(25, 100) #Don't uncomment, just to show what each column (particle) should evaluate to for sphere
        ))

        # Array to be updated.  The left column should be better (0 vs 25) and update, but the right column
        # Is better in the past (100 v 200)
        past_p_best = np.array((
            (11, 12),
            (13, 14),
            (0, 200,)
        ))

        # What the array should look like
        p_best_expected = np.array((
            (11, 6),
            (13, 8),
            (0, 100)
        ))

        p_best_test = update.update_p_best(pos_matrix, past_p_best, sphere_func)
        np.testing.assert_array_almost_equal(p_best_test, p_best_expected)

    def test_update_velocity(self):
        """Check to make sure that the update_velocity function works as intended"""
        v_part_expected = np.array((
            (1, 2),
            (3, 4)
        ))
        x_pos = np.array((
            (5, 6),
            (7, 8)
        ))
        p_best = np.array((
            (-1, -2),
            (-1, -2),
            (-1, -2)
        ))
        g_best = np.array((-2, -2, -2))

        w = 0.5
        c1 = 0.3
        c2 = 0.4

        v_part_test = update.update_velocity(
            v_part=v_part_expected, 
            x_pos=x_pos, 
            p_best = p_best, 
            g_best = g_best, 
            w = w, 
            c1 = c1,
            c2 = c2
        )

        # Velocity matrix attained from experimental run when seed = 0
        v_part_expected = np.array((
            (-2.17560176, -2.46008066),
            (-1.98710056, -2.32510083)
        ))
        np.testing.assert_array_almost_equal(v_part_test, v_part_expected, 8)

    def test_update_position(self):
        """Check to make sure update_position function works as intended"""
        v_part = np.array((
            (1, 2),
            (3, 4)
        ))

        x_pos = np.array((
            (5, 6),
            (7, 8)
        ))

        x_pos_test = update.update_position(x_pos, v_part)

        # Expected x pos is just v_part + x_pos
        expected_x_pos = np.array((
            (6, 8),
            (10, 12)
        ))
        np.testing.assert_array_almost_equal(x_pos_test, expected_x_pos)

    def test_verify_bounds(self):
        """Check to make sure that the verify_bounds function works as intended"""
        upper_bounds = np.ones((2))*5
        lower_bounds = np.ones((2))*-6

        # 10 (higher than 5) is out of bounds and -13 is out of bounds (lower than -6).
        matrix = np.array((
            (10, 4),
            (-13, 0)
        ))

        # The above matrix should look like this after verify_bounds is called.
        expected_matrix = np.array((
            (5, 4),
            (-6, 0)
        ))

        matrix_test = update.verify_bounds(matrix=matrix, upper_bounds=upper_bounds, lower_bounds=lower_bounds)
        np.testing.assert_array_almost_equal(matrix_test, expected_matrix)

class CCDTester(PSOTester):
    """Class which tests the ccd function"""
    def test_ccd(self):
        dim = 30
        ub = np.ones(dim)*100
        lb = -1*ub
        initial = util.scale(lb=lb, ub=ub, array=np.random.rand(dim))
        optimum = np.zeros(dim)
        initial = np.append(initial, 0)
        test_func = tf.TestFuncts.generate_function("rosenbrock", optimum, bias=0)
        ccd_test = ccd.CCD(initial, lb, ub, alpha = 0.2, tol=0.0001, max_its = 20, third_term_its = 6, func = test_func)
        
        # Experimental ccd g_best, when ccd was 0
        ccd_expected = np.array((7.51128284e-10, 1.46210307e-09, 2.78059842e-09, 5.41143455e-09,
            1.09594394e-08, 2.21391735e-08, 4.44186325e-08, 8.91780998e-08,
            1.78776744e-07, 3.58512067e-07, 7.18774870e-07, 1.44131185e-06,
            2.89022089e-06, 5.79582528e-06, 1.16225588e-05, 2.33073921e-05,
            4.67399175e-05, 9.37316931e-05, 1.87972933e-04, 3.76985597e-04,
            7.56127576e-04, 1.51686804e-03, 3.04414348e-03, 6.11381647e-03,
            1.22976337e-02, 2.48118927e-02, 5.03702848e-02, 1.03538553e-01,
            2.18312178e-01, 4.84284570e-01, 6.17697642e-02))
        
        np.testing.assert_array_almost_equal(ccd_test, ccd_expected, 8)
    
if __name__ == "__main__":
    unittest.main()