import unittest
import numpy as np
from latools.processes import *


class test_process_functions(unittest.TestCase):

    def test_noise_despike(self):
        test_data = np.array([2000, 2000, 20000, 2000, 2000])
        target = np.array([2000, 2000, 2666, 2000, 2000])

        out = noise_despike(test_data)

        self.assertTrue(all(out == target))


if __name__ == '__main__':
    unittest.main()
