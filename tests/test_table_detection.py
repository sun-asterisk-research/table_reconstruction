import unittest

import cv2
import numpy as np
import torch
import os
import shutil

from table_reconstruction.table_detection.preprocess import create_batch, process_image
from table_reconstruction.table_detection.detector import MODEL_PATH, TableDetector
from table_reconstruction.table_detection.utils import (
    DETECTOR_WEIGHT_URL,
    download_weight,
    select_device,
)


class PreprocessTestCase(unittest.TestCase):
    def test_create_batch(self):
        images = [
            np.random.randint(255, size=(640, 530, 3), dtype=np.uint8),
            np.random.randint(255, size=(640, 300, 3), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 3), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 3), dtype=np.uint8),
            np.random.randint(255, size=(640, 300, 3), dtype=np.uint8),
            np.random.randint(255, size=(640, 530, 3), dtype=np.uint8),
        ]

        images_batch, indices = create_batch(
            images, set(list(x.shape for x in images)), batch_size=2
        )
        self.assertEqual(sum(len(batch) for batch in images_batch), len(images))
        self.assertListEqual([0, 2, 3, 5, 1, 4], indices)

    def test_process_image(self):
        image = np.random.randint(255, size=(640, 530, 3), dtype=np.uint8)
        output = process_image(image)
        self.assertFalse(output.is_cuda)
        self.assertListEqual(list(output.shape), [3, 640, 640])
        self.assertTrue(output.dtype == torch.float32)


class TableDetectionTestCase(unittest.TestCase):
    def test_load_model(self):
        parent_dir = os.path.dirname(MODEL_PATH)
        if os.path.exists(parent_dir):
            shutil.rmtree(parent_dir)
        # invalid weights file path
        with self.assertRaises(Exception):
            TableDetector(model_path="/tmp/test.pt")
        # Download  weights file
        TableDetector()
        # valid weights file path
        self.assertTrue(hasattr(TableDetector(MODEL_PATH), "model"))

    def test_predict(self):
        image = cv2.imread("tests/tableImg.jpg")
        fake_img = np.ones((640, 640, 3), dtype=np.uint8)
        predicted = TableDetector().predict([image])
        fake_predicted = TableDetector().predict([fake_img])
        self.assertTupleEqual(np.array(predicted).shape, (1, 1, 4))
        self.assertTupleEqual(np.array(fake_predicted).shape, (1, 0))


class TestTableUtils(unittest.TestCase):
    def test_download_weights(self):
        self.assertEqual(download_weight(DETECTOR_WEIGHT_URL, "test.pt"), "test.pt")

    def test_select_device(self):
        self.assertEqual(select_device(torch.device("cpu")), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
