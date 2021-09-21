from typing import Any, Union

import torch


def _create(
    name,
    pretrained: bool = True,
    channels: int = 3,
    classes: int = 80,
    autoshape: bool = True,
    verbose: bool = True,
    device: Union[str, torch.device] = None,
):
    """Creates a specified YOLOv5 model
    Arguments:
        name (str): name of model, i.e. 'yolov5s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes
        autoshape (bool): apply YOLOv5 .autoshape() wrapper to model
        verbose (bool): print all information to screen
        device (str, torch.device, None): device to use for model parameters
    Returns:
        YOLOv5 pytorch model
    """
    from pathlib import Path

    from models.experimental import attempt_load
    from models.yolo import Model

    file = Path(__file__).absolute()

    save_dir = Path("") if str(name).endswith(".pt") else file.parent
    path = (save_dir / name).with_suffix(".pt")  # checkpoint path
    try:
        device = select_device(
            ("0" if torch.cuda.is_available() else "cpu") if device is None else device
        )

        if pretrained and channels == 3 and classes == 80:
            model = attempt_load(path, map_location=device)  # download/load FP32 model
        else:
            cfg = list((Path(__file__).parent / "models").rglob(f"{name}.yaml"))[
                0
            ]  # model.yaml path
            model = Model(cfg, channels, classes)  # create model
        if autoshape:
            model = model.autoshape()  # for file/URI/PIL/cv2/np inputs and NMS
        return model.to(device)

    except Exception as e:
        help_url = "https://github.com/ultralytics/yolov5/issues/36"
        s = (
            "Cache may be out of date, try `force_reload=True`. See %s for help."
            % help_url
        )
        raise Exception(s) from e


def custom(path="path/to/model.pt", autoshape=True, verbose=True, device=None):
    # YOLOv5 custom or local model
    return _create(path, autoshape=autoshape, verbose=verbose, device=device)


def select_device(device: Union[str, Any] = "") -> torch.device:
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
