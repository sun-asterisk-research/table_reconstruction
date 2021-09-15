import torch

from .unet.resunet import ResUnet


def load_model_unet(weight_path: str, device: torch.device) -> torch.nn.Module:
    """Initialize the line segment model

    Args:
        weight_path (str): path to weight file
        device (torch.device): torch device

    Returns:
        torch.nn.Module: Unet model
    """
    net = ResUnet(n_classes=1)
    net.to(device=device)
    # load pretrained weight
    net.load_state_dict(torch.load(weight_path, map_location=device))
    return net
