import unittest
from . import initializer, update
import logging
import numpy as np
import testfuncts as tf

class PSOTester(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)
    seed_int = 0

    def reset_seed(self) -> None:
        np.random.seed(self.seed_int)

    def setUp(self) -> None:
        self.reset_seed()
        return super().setUp()

class PSOInitializser(PSOTester):

    def test_x_initializer(self):
        """Test to make sure tha the position matrix initializes correctly"""
        # Test to make sure initialization works
        num_dim = 2
        num_part = 2
        upper_bound = np.ones((num_dim))*5
        lower_bound = np.ones((num_dim))*-4

        pos_matrix = np.array(((0.93932154, 2.4367043), (1.42487038, 0.90394865)))
        pos_matrix_test = initializer._x_initializer(num_dim, num_part, upper_bound, lower_bound)

        # Test to make sure that the position matrix matches what we get experimentally for the 0 seed
        np.testing.assert_array_almost_equal(pos_matrix, pos_matrix_test, decimal=8)

    def test_v_initializer(self):
        """Test to make sure that the velocity and vmax initializes correctly"""
        num_dim = 2
        num_part = 2
        upper_bound = np.ones((num_dim))*6
        lower_bound = np.ones((num_dim))*-7
        alpha = 0.3

        v_matrix = np.array(((0.38074533, 1.67847706), (0.80155433, 0.35008883)))
        v_max = np.array((3.9, 3.9))

        v_matrix_test, v_max_test = initializer._v_initializer(num_dim, num_part, upper_bound, lower_bound, alpha)
        # Test to make sure that the velocity and vmax matrices match what we get experimentally for the 0 seed
        np.testing.assert_array_almost_equal(v_matrix, v_matrix_test, 8)
        np.testing.assert_array_almost_equal(v_max, v_max_test, 8)

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

        pos_matrix = np.array(((0.23220256, 2.7278405), (1.04145064, 0.17324774)))
        vel_matrix = np.array(((-0.4580712 ,  0.87536468), (-0.37447673,  2.350638)))
        v_max = np.array((3, 3))

        p_best_val = []

        min_val = np.finfo(np.float64).max
        min_ind = 0

        for i in range(pos_matrix.shape[1]):
            val = sphere_func(pos_matrix[:, i])
            if val < min_val:
                min_ind = i
                min_val = val

            p_best_val.append(val)
        p_best_val = np.array(p_best_val)
        p_best = np.vstack((pos_matrix, p_best_val))

        g_best = p_best[:, min_ind]

        # Check to make sure all of the returned values match what we expect
        np.testing.assert_array_almost_equal(pos_matrix_test, pos_matrix, 8)
        np.testing.assert_array_almost_equal(vel_matrix_test, vel_matrix, 8)
        np.testing.assert_array_almost_equal(v_max_test, v_max, 8)
        np.testing.assert_array_almost_equal(p_best_test, p_best, 7)
        np.testing.assert_array_almost_equal(g_best_test, g_best, 8)

class PSOUpdater(PSOTester):
    def test_update_g_best(self):
        test_p_best = np.array((
            (1, 2),
            (3, 4),
            (0, 1)
        ))

        g_best = test_p_best[:, 0]

        g_best_test = update.update_g_best(test_p_best)
        np.testing.assert_array_almost_equal(g_best_test, g_best, 8)

        # Test to make sure that the returned array does not interfere with the pbest
        # Will fail if g_best is treated as a pointer and not an independent array
        test_p_best[0, 0] += 1
        self.assertNotAlmostEqual(test_p_best[0, 0], g_best_test[0])

    def test_update_p_best(self):
        optimum = np.zeros((2))

        sphere_func = tf.TestFuncts.generate_function("sphere", optimum=optimum, bias = 0)

        # Position matrix, which for sphere, should evaluate to 25 and 100 for each respective column
        pos_matrix = np.array((
            (3, 6),
            (4, 8)
            #(25, 100) #Don't uncomment, just to show what it should evaluate to for sphere
        ))

        # Array to be updated.  The left column should be better (6 vs 5) and update, but the right column
        # Is better in the past (9 v 10)
        past_p_best = np.array((
            (11, 12),
            (13, 14),
            (0, 200,)
        ))

        # What the array should look like
        p_best = np.array((
            (11, 6),
            (13, 8),
            (0, 100)
        ))

        p_best_test = update.update_p_best(pos_matrix, past_p_best, sphere_func)
        np.testing.assert_array_almost_equal(p_best_test, p_best)

    def test_update_velocity(self):
        v_part = np.array((
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
            v_part=v_part, 
            x_pos=x_pos, 
            p_best = p_best, 
            g_best = g_best, 
            w = w, 
            c1 = c1,
            c2 = c2
        )

        v_part = np.array((
            (-2.17560176, -2.24599521),
            (-2.17803394, -2.32510083)
        ))

        np.testing.assert_array_almost_equal(v_part_test, v_part, 8)

    def test_update_position(self):
        v_part = np.array((
            (1, 2),
            (3, 4)
        ))

        x_pos = np.array((
            (5, 6),
            (7, 8)
        ))

        x_pos_test = update.update_position(x_pos, v_part)

        expected_x_pos = np.array((
            (6, 8),
            (10, 12)
        ))

        np.testing.assert_array_almost_equal(x_pos_test, expected_x_pos)

    def test_verify_bounds(self):
        upper_bounds = np.ones((2))*5
        lower_bounds = np.ones((2))*-6

        matrix = np.array((
            (10, 4),
            (-13, 0)
        ))

        expected_matrix = np.array((
            (5, 4),
            (-6, 0)
        ))

        matrix_test = update.verify_bounds(matrix=matrix, upper_bounds=upper_bounds, lower_bounds=lower_bounds)

        np.testing.assert_array_almost_equal(matrix_test, expected_matrix)

    
if __name__ == "__main__":
    unittest.main()