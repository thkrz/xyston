import numpy as np
import torch
import unittest

import xyston.nn as xy


class Test(unittest.TestCase):
    def test_max_pool(self):
        x = torch.tensor(np.random.rand(1, 1, 4, 4, 4, 4))
        p = xy.MaxPool4d((2, 2, 2, 2))
        y = p(x)
        print(y)


if __name__ == "__main__":
    unittest.main()
