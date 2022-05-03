import numpy as np
import torch
import torch.nn as nn
import unittest

import xyston.nn as xy


class Test(unittest.TestCase):
    def test_max_pool(self):
        x = torch.tensor(np.random.rand(1, 2, 1, 4, 4, 4, 4))
        p = xy.CMaxPool4d(2)
        y = p(x)
        print(y)
        print(y.shape)


if __name__ == "__main__":
    unittest.main()
