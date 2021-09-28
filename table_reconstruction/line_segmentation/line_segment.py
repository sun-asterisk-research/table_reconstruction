import logging
import os
from typing import Tuple

import cv2
import gdown
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .utils import load_model_unet

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = DIR_PATH + "/tmp/resnet_weight.pth"
WEIGHT_URL = "https://drive.google.com/u/0/uc?id=18YEiAzUs9NXz0FwBuU0JicEWc_F2V7tq"


class LineSegmentation:
    def __init__(
        self,
        model_path: str = None,
        device: torch.device = torch.device("cpu"),
    ):
        """The constructor of class LineSegmentation which creates object
        for segmenting lines in table images

        Args:
            model_path (str): path to model weight file
            device (torch.device, optional): torch device. Defaults is "cpu".
        """
        self.device = device
        if model_path is None:
            if not os.path.exists(MODEL_PATH):
                logging.info("Obtain weight of model...")
                if not os.path.exists(DIR_PATH + "/tmp/"):
                    os.mkdir(DIR_PATH + "/tmp/")
                try:
                    gdown.download(url=WEIGHT_URL, output=MODEL_PATH, quiet=False)
                except Exception as e:
                    logging.info("Could not download weight, please try again!")
                    logging.info(f"Error code: {e}")
                    raise Exception("An error occured while downloading weight file")
            self.model = load_model_unet(MODEL_PATH, device)
        else:
            if os.path.exists(model_path):
                self.model = load_model_unet(model_path, device)
            else:
                raise ValueError(f"Could not find weight file at {model_path}")

    def predict(
        self,
        img: np.ndarray,
        scale_factor: float = 0.5,
        out_threshold: float = 0.5,
    ) -> np.ndarray:
        """Take input as an table image and return a mask of the same size
        as the input image and each pixel has a value of 1 if that pixel belongs
        to a line otherwise it will be 0.

        Args:
            img (np.array): table image
            scale_factor (float, optional): factor for downscaling original image.
            Defaults to 0.5.
            out_threshold (float, optional): confidence threshold. Defaults to 0.5.

        Returns:
            numpy.ndarray: mask image has same size with original image
        """
        self.model.eval()
        padding_pil_img, preprocessed_img, pad = self._preprocess(
            img=img,
            scale=scale_factor,
        )
        ts_img = torch.from_numpy(preprocessed_img)
        ts_img = ts_img.unsqueeze(0)
        ts_img = ts_img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(ts_img)
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
            mask = self._normalize(padding_pil_img, mask_img=mask)
            mask = np.array(mask[pad:-pad, pad:-pad])
            return mask

    def _preprocess(
        self,
        img: np.ndarray,
        scale: float,
        pad: int = 5,
    ) -> Tuple[Image.Image, np.ndarray, int]:
        """Add pad to table image from path then resize image

        Args:
            img (np.array): table image
            scale (float): Scale factor
            pad (int, optional): Pad to add to image. Defaults to 5.

        Returns:
            PIL.Image.Image: PIL image for size-recovering purpose
            numpy.array: image after preprocessing
            int: pad value for size-recovering purpose
        """
        # Padding
        h, w, _ = img.shape
        assert pad >= 0, "Pad must great than 0"
        padding_img = np.ones((h + pad * 2, w + pad * 2, 3), dtype=np.uint8) * 255
        padding_img[pad : h + pad, pad : w + pad, :] = img
        pil_img = Image.fromarray(padding_img)

        # Resize
        newW, newH = int(scale * (w + pad * 2)), int(scale * (h + pad * 2))
        assert newW > 0 and newH > 0, "Scale is too small"
        rz_pil_img = pil_img.resize((newW, newH))
        img_nd = np.array(rz_pil_img)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return pil_img, img_trans, pad

    def _normalize(self, img: Image.Image, mask_img: np.ndarray) -> np.ndarray:
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
