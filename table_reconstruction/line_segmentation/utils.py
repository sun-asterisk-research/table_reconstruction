import torch

from .unet.repunet import create_RepVGG_A0
from .unet.resunet import ResUnet


def load_model_unet(
    weight_path: str,
    device: torch.device,
    use_vgg=True,
) -> torch.nn.Module:
    """Initialize the line segment model

    Args:
        weight_path (str): path to weight file
        device (torch.device): torch device
        use_vgg (bool, optional): using RepVGG based model or not. Defaults to True.

    Returns:
        torch.nn.Module: Unet model
    """
    if use_vgg:
        net = create_RepVGG_A0(True)
    else:
        net = ResUnet(n_channels=3, n_classes=1)

    net.to(device=device)

    # load pretrained weight
    net.load_state_dict(torch.load(weight_path, map_location=device))

    return net
