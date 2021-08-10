import os
import os
import shutil
import unittest

import numpy as np
import pandas as pd
from ddt import ddt, data

from ..utils import combine_statistics


def generate_test_data(inputpath, dlen, nfiles):
    for i in range(nfiles):
        arr = np.random.randint(0, 100, [dlen, 3])
        stat = pd.DataFrame(arr, columns=['a', 'b', 'c'])
        fn = os.path.join(inputpath, "test/" * i, 'data.csv')
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        stat.to_csv(fn, index=False)


@ddt
class TestUtils(unittest.TestCase):

    @data(
        (2, 10),
        (100, 3)
    )
    def test_stat(self, case):
        dlen, nfiles = case
        path = 'test'
        generate_test_data(path, dlen, nfiles)
        combine_statistics(path)
        stat = pd.read_csv(path + '.csv')
        self.assertEqual(len(stat), dlen * nfiles)
        shutil.rmtree(path)
        os.remove(path + '.csv')


if __name__ == '__main__':
    unittest.main()
