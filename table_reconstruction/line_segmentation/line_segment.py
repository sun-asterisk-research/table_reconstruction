import logging
import os

from typing import Tuple

import gdown
import torch
import numpy as np
import cv2

from torchvision import transforms
from PIL import Image

from .dataset import BasicDataset
from .utils import load_model_unet


MODEL_PATH = os.path.abspath("tmp/repvgg_weight_v3.pth")
WEIGHT_URL = "https://drive.google.com/u/1/uc?id=16atStSuFjpgwX54E2bvT6q8LF_1MyjiY"


class LineDetector:
    def __init__(self, model_path: str = MODEL_PATH, device: str = "cpu"):
        self.device = device
        if not os.path.exists(MODEL_PATH):
            logging.info("Downloading weight of model...")
            if not os.path.exists(os.path.abspath("tmp/")):
                os.mkdir(os.path.abspath("tmp/"))
            gdown.download(url=WEIGHT_URL, output=MODEL_PATH, quiet=False)
        self.model = load_model_unet(model_path, device)

    def predict(
        self, img_path: str, scale_factor: float = 0.5, out_threshold: float = 0.5
    ) -> np.ndarray:
        """Take input as an table image and return a mask of the same size
        as the input image and each pixel has a value of 1 if that pixel belongs
        to a line otherwise it will be 0.

        Args:
            img_path (str): path to table image
            scale_factor (float, optional): factor for dowscaling original image. Defaults to 0.5.
            out_threshold (float, optional): confidence threshold. Defaults to 0.5.

        Returns:
            numpy.ndarray: mask image has same size with original image
        """
        self.model.eval()
        preprocessed_img, pad = self.preprocess_img(img_path)
        img = torch.from_numpy(BasicDataset.preprocess(preprocessed_img, scale_factor))
        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img)
            probs = torch.sigmoid(output)
            probs = probs.squeeze(0)
            tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
            )

            probs = tf(probs.cpu())
            full_mask = probs.squeeze().cpu().numpy()
            mask = full_mask > out_threshold
            mask = self.normalize(preprocessed_img, mask_img=mask)
            mask = np.array(mask[pad:-pad, pad:-pad])
            return mask

    def preprocess_img(self, img_path: str, pad: int = 5) -> Tuple[Image.Image, int]:
        """Add pad to table image from path

        Args:
            img_path (str): Path to table image
            pad (int, optional): Pad to add to image. Defaults to 5.

        Returns:
            PIL.Image.Image: image after padding
            int: pad value for size-recovering purpose
        """
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        padding_img = np.ones((h + pad * 2, w + pad * 2, 3), dtype=np.uint8) * 255
        padding_img[pad:h + pad, pad:w + pad, :] = img
        preprocessed_img = Image.fromarray(padding_img)
        return preprocessed_img, pad

    def normalize(self, img: Image.Image, mask_img: np.ndarray) -> np.ndarray:
        """Convert shape of mask image to shape of img

        Args:
            img (PIL.Image.Image): original table image (H, W, C)
            mask_img (numpy.ndarray): binary image of original table image (H1, W1, C1)

        Returns:
            numpy.ndarray: Mask image has shape of (H, W, C)
        """
        mask = np.asarray(mask_img)
        img = np.asarray(img)
        img_h, img_w = img.shape[:2]
        mask = mask.reshape(mask.shape[0], mask.shape[1])
        mask = mask.astype(np.uint8)
        mask = cv2.resize(mask, (img_w, img_h), cv2.INTER_AREA)

        return mask
