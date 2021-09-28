import unittest
import numpy as np
import os
import shutil

import table_reconstruction as tr
from table_reconstruction.line_segmentation.line_segment import MODEL_PATH


class TestLineSegmentation(unittest.TestCase):
    def test_load_model(self):
        # invalid weight file path
        with self.assertRaises(ValueError):
            tr.LineSegmentation("/tmp/resnet_weight.pth")
        # Download weight file
        tr.LineSegmentation()
        # valid weight file path
        self.assertTrue(hasattr(tr.LineSegmentation(MODEL_PATH), "model"))

    def test_mask_size(self):
        random_array = np.ones((200, 200, 3))
        mask = tr.LineSegmentation().predict(random_array)
        parent_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(parent_dir):
            shutil.rmtree(parent_dir)
        self.assertEqual(random_array.shape[0:2], mask.shape[0:2])


if __name__ == "__main__":
    unittest.main()
