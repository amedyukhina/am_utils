import os
import shutil
import unittest

import numpy as np
from ddt import ddt, data
from skimage import io

from ..imgproc import convert_to_isotropic, convert_to_isotropic_batch
from ..utils import imsave, walk_dir


@ddt
class TestIsotropicConversion(unittest.TestCase):
    @data(
        np.ones([10] * 3),
        np.random.randint(0, 100, [10] * 3)
    )
    def test_valid_shape(self, img):
        output = convert_to_isotropic(img, [0.2, 0.1, 0.1])
        self.assertEqual(len(output.shape), len(img.shape))

    @data(
        np.ones([10] * 1),
        np.random.randint(0, 100, [10] * 2),
        np.random.randint(0, 100, [10] * 4)
    )
    def test_invalid_shape(self, img):
        self.assertRaises(ValueError, convert_to_isotropic, img, [0.2, 0.1, 0.1])

    @data(
        np.uint8,
        np.uint16,
        np.float32
    )
    def test_batch(self, imgtype):
        inputpath = 'test_input'
        outputpath = 'test_output'
        for i in range(3):
            img = np.random.randint(0, 100, [10] * 3)
            img = img.astype(imgtype)
            imsave(os.path.join(inputpath, "test/" * i, 'img.tif'), img)

        convert_to_isotropic_batch(inputpath, outputpath,
                                   n_jobs=8, verbose=False,
                                   voxel_size=[5, 1, 1])
        self.assertEqual(len(walk_dir(inputpath)), len(walk_dir(outputpath)))
        imgout = io.imread(walk_dir(outputpath)[0])
        self.assertEqual(type(imgout[0, 0, 0]), imgtype)
        self.assertEqual(imgout.shape[0], 10*5)
        shutil.rmtree(inputpath)
        shutil.rmtree(outputpath)


if __name__ == '__main__':
    unittest.main()
