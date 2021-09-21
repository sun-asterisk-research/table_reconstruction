from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch


def create_batch(
    images: List[np.ndarray], shapes: set, batch_size: int = 16
) -> Tuple[list, list]:
    """Create batch image with same shape for inference
        Return list batch image

    Args:
        images (np.ndarray): List all input images
        shapes (set): set all shapes of input images
        batch_size (int, optional): number images in a batch. Defaults to 16.

    Returns:
        [type]: [description]
        images_batch (list): list batch images
        indices (list): order of all input image
    """
    split_batch = []
    images_batch = []
    for shape in shapes:
        mini_batch = []
        images_mini_batch: List[Any] = []
        for idx, img in enumerate(images):
            if img.shape == shape:
                mini_batch.append(idx)
                if len(images_mini_batch) < batch_size:
                    images_mini_batch.append(img)
                else:
                    images_batch.append(images_mini_batch)
                    images_mini_batch = []
                    images_mini_batch.append(img)
        images_batch.append(images_mini_batch)
        split_batch.append(mini_batch)
    del images_mini_batch

    indices = [item for sublist in split_batch for item in sublist]
    return images_batch, indices


def process_image(img: Union[np.ndarray, torch.Tensor], device="cpu") -> torch.Tensor:
    """preprocess image before inference
        image will be resized, crop add border and convert to Torch Tensor

    Args:
        img (np.ndarray): input image
        device (str, optional): device where input image is
        allocated on. Defaults to "cpu".

    Returns:
        [type]: [description]
        img (Torch.tensor): image after perprocessing
    """
    height, width = img.shape[:2]
    top = (640 - height) // 2
    bottom = 640 - height - top
    left = (640 - width) // 2
    right = 640 - width - left
    img = cv2.copyMakeBorder(
        img,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = torch.transpose(img, 0, 1)
    img = torch.transpose(img, 0, 2)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0
    return img
