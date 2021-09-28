import cv2
import numpy as np


def normalize(img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
    """Convert shape of mask image to shape of input image

    Args:
        img (np.ndarray): input image has shape of (H, W, C)
        mask_img (np.ndarray): binary image of original image has shape (H1, W1, C1)

    Returns:
        np.ndarray: binary image of original image has shape (H, W, C)
    """
    mask = np.asarray(mask_img)
    img = np.asarray(img)
    img_h, img_w = img.shape[:2]
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (img_w, img_h), cv2.INTER_AREA)

    return mask


def get_hor_lines_mask(mask: np.ndarray, kernel_len: int) -> np.ndarray:

    """Get binary image which contain only horizontal lines

    Args:
        mask (np.ndarray): binary image of original image
        kernel_len (int): kernel size in cv2.getStructuringElement method

    Returns:
        np.ndarray: binary image contain only horizontal lines
    """
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    image_horizontal = cv2.erode(mask, hor_kernel, iterations=2)
    horizontal_lines_mask = cv2.dilate(image_horizontal, hor_kernel, iterations=3)

    return horizontal_lines_mask


def get_ver_lines_mask(mask: np.ndarray, kernel_len: int) -> np.ndarray:

    """Get binary image which contain only vertical lines

    Args:
        mask (np.ndarray): the binary image of table (H, W, C)
        kernel_len (int): kernel size in cv2.getStructuringElement method

    Returns:
        np.ndarray: binary image contain only vertical lines
    """
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    image_vertical = cv2.erode(mask, ver_kernel, iterations=2)
    vertical_lines_mask = cv2.dilate(image_vertical, ver_kernel, iterations=3)

    return vertical_lines_mask
