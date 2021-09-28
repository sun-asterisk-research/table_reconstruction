import logging
import os
from functools import partial
from typing import List

import numpy as np
import torch

from .yolov5.models.utils import non_max_suppression, scale_coords
from .preprocess import create_batch, process_image
from .utils import DETECTOR_WEIGHT_URL, download_weight, load_yolo_model, select_device
from .yolov5.models.utils import letterbox

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = DIR_PATH + "/tmp/model_table_detect.pt"


class TableDetector:
    def __init__(
        self,
        model_path: str = None,
        device: str = "cpu",
        conf_thres: float = 0.85,
        iou_thres: float = 0.3,
    ):
        """The constructor of class TableDetector which creates object
            for detcting table regions in pdf page images

        Args:
            model_path (str, optional): Path to weights file. Defaults to None.
            device (str, optional): Device where model will be allocated on.
            Defaults to "cpu".
            conf_thres (float, optional): Confidence threshold. Defaults to 0.85.
            iou_thres (float, optional): IoU threshold. Defaults to 0.3.
        """

        # Run first time when using
        if model_path is None:
            if not os.path.exists(MODEL_PATH):
                logging.info("Obtain weights of model ...")
                if not os.path.exists(DIR_PATH + "/tmp/"):
                    os.mkdir(DIR_PATH + "/tmp/")
                try:
                    logging.info("Downloading weight from google drive")
                    download_weight(DETECTOR_WEIGHT_URL, output=MODEL_PATH),
                except Exception as e:
                    logging.info("Could not download weight, please try again!")
                    logging.info(f"Error code: {e}")
                    raise Exception("An error occured while downloading weight file")
            self.model, self.stride = load_yolo_model(MODEL_PATH, device=device)
        else:
            if os.path.exists(model_path):
                self.model, self.stride = load_yolo_model(model_path, device=device)
            else:
                raise ValueError(f"Could not find weights file at {model_path}")

        self.device = select_device(device)
        logging.info("Using {} for table detection".format(self.device))

        self.img_size = 640
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.process_func_ = partial(process_image, device=self.device)

    def predict(self, image_list: List[np.ndarray]) -> List[list]:
        """
        Returns a list of bounding boxes [xmin, ymin, xmax, ymax] for
        each image in image_list
        Each element in the list is a numpy array of shape N x 4

        Args:
            image_list (List[np.array]): input images

        Returns:
            [List[list]]: output bounding boxes
        """
        batches, indices = create_batch(
            image_list, set(list(x.shape for x in image_list))
        )
        predictions = []

        for origin_images in batches:
            images = [letterbox(x, 640, stride=32)[0] for x in origin_images]
            images = list(map(self.process_func_, images))
            tensor = torch.stack(images).to(device=self.device)

            with torch.no_grad():
                pred = self.model(tensor)[0]
            all_boxes = []
            pred = non_max_suppression(pred, 0.3, 0.30, classes=0, agnostic=True)

            for idx, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(
                        images[idx].shape[1:], det[:, :4], origin_images[0].shape
                    ).round()
                    det = det[:, :4]
                    all_boxes.append(det.cpu().numpy().astype("int").tolist())
                else:
                    all_boxes.append([])

            predictions.extend(all_boxes)

        z = zip(predictions, indices)
        sorted_result = sorted(z, key=lambda x: x[1])
        predictions, _ = zip(*sorted_result)

        return list(predictions)
