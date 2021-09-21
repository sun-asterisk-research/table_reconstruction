from typing import Union
import gdown
import torch

from .yolov5 import YOLO_DIR

DETECTOR_WEIGHT_URL = "https://drive.google.com/uc?id=12ttln8zPOWrFCPLr4hChmxr4rxuHRRoz"


def select_device(device: str = "") -> torch.device:
    """Select device where model and image will be allocated on

    Args:
        device (str, optional): name of device (cpu, cuda, ..). Defaults to "".

    Returns:
        device (torch.device): selected device
    """
    if not isinstance(device, str):
        device = device.type
    cpu = device.lower() == "cpu"
    cuda = not cpu and torch.cuda.is_available()
    return torch.device("cuda:0" if cuda else "cpu")


def load_yolo_model(weight_path: str, device: Union[torch.device, str]):
    """load yolo model detect using torch hub

    Args:
        weight_path (str): path to weights file
        device (str): name of the deivce where model will be allocated on

    Returns:
        model (models.common.autoShape): model load with torch hub
        model stride (torch.Tensor): stride of model
    """
    model = torch.hub.load(
        str(YOLO_DIR),
        "custom",
        path=weight_path,
        source="local",
        device=device,
        force_reload=True,
    )
    if isinstance(device, str):
        device = torch.device(device)
    model.to(device)
    return model, model.stride


def download_weight(url: str, output=None, quiet=False) -> str:
    """Download model weights from google drive using gdown

    Args:
        url (str): Google drive direct downloaded link of model weights file
        output ([type], optional):  Ouput filename. Defaults to None.
        quiet (bool, optional): Suppress terminal output. Default is False

    Returns:
        name of the model weights (str)
    """
    return gdown.download(url=url, output=output, quiet=quiet)
