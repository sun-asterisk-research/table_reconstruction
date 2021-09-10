import re

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.hook import dummy_eval, hook_outputs, model_sizes
from fastai.layers import (
    BatchNorm,
    ConvLayer,
    MergeLayer,
    PixelShuffle_ICNR,
    ResBlock,
    SigmoidRange,
)
from fastai.torch_core import apply_init
from torch import nn
from torchvision.models import resnet18


def check_pool_layer(layer):
    return re.search(r"Pool[123]d$", layer.__class__.__name__)


def get_sz_change_idxs(sizes):
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sz_chg_idxs = list(
        np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0]
    )
    return sz_chg_idxs


def has_pool_layer(m):
    "Return `True` if `m` is a pooling layer or has one in its children"
    if check_pool_layer(m):
        return True
    for layer in m.children():
        if has_pool_layer(layer):
            return True
    return False


def create_body(net):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    ll = list(enumerate(net.children()))
    cut = next(i for i, o in reversed(ll) if has_pool_layer(o))
    return nn.Sequential(*list(net.children())[:cut])


class SequentialExtend(nn.Module):
    "Like `nn.Sequential`, but with ModuleList semantics, and can access module input"

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for layer in self.layers:
            res.orig = x
            nres = layer(res)
            res.orig, nres.orig = None, None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, layer):
        return self.layers.append(layer)

    def extend(self, layer):
        return self.layers.extend(layer)

    def insert(self, i, layer):
        return self.layers.insert(i, layer)


class UBlock(nn.Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(
        self,
        up_in_c,
        x_in_c,
        hook,
        final_div=True,
        blur=False,
        act_cls=nn.ReLU,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        **kwargs
    ):
        super().__init__()
        self.hook = hook
        self.shuf = PixelShuffle_ICNR(
            up_in_c, up_in_c // 2, blur=blur, act_cls=act_cls, norm_type=norm_type
        )
        self.bn = BatchNorm(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        # nf = ni // 2
        self.conv1 = ConvLayer(ni, nf, act_cls=act_cls, norm_type=norm_type, **kwargs)
        self.relu = act_cls()
        apply_init(self.conv1, init)

    def forward(self, up_in):
        s = self.hook.stored
        up_out = forward_u_block(s, self.shuf(up_in))
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv1(cat_x)


@torch.jit.script
def forward_u_block(s, up_out):

    ssh = s.shape[-2:]
    if ssh != up_out.shape[-2:]:
        up_out = F.interpolate(up_out, s.shape[-2:], mode="nearest")
    return up_out


class ResizeToOrig(nn.Module):
    def __init__(self, mode="nearest"):
        super().__init__()
        self.mode = mode

    def forward(self, x):
        return forward_resize(x.orig, x)


@torch.jit.script
def forward_resize(origin, tensor):
    if origin.shape[-2:] != tensor.shape[-2:]:
        tensor = F.interpolate(tensor, origin.shape[-2:], mode="nearest")
    return tensor


class DynamicUnet(SequentialExtend):
    "Create a U-Net from a given architecture."

    def __init__(
        self,
        encoder,
        n_out,
        img_size,
        blur=False,
        blur_final=True,
        self_attention=False,
        y_range=None,
        last_cross=True,
        bottle=False,
        act_cls=nn.ReLU,
        init=nn.init.kaiming_normal_,
        norm_type=None,
        **kwargs
    ):
        imsize = img_size
        sizes = model_sizes(encoder, size=imsize)
        sz_chg_idxs = list(reversed(get_sz_change_idxs(sizes)))
        self.sfs = hook_outputs([encoder[i] for i in sz_chg_idxs], detach=False)
        x = dummy_eval(encoder, imsize).detach()

        ni = sizes[-1][1]
        middle_conv = ConvLayer(
            ni, ni // 2, act_cls=act_cls, norm_type=norm_type, **kwargs
        ).eval()
        x = middle_conv(x)

        layers = [encoder, BatchNorm(ni), nn.ReLU(), middle_conv]
        ni = ni // 2
        for i, idx in enumerate(sz_chg_idxs):
            not_final = i != len(sz_chg_idxs) - 1
            up_in_c, x_in_c = int(x.shape[1]), int(sizes[idx][1])
            do_blur = blur and (not_final or blur_final)
            sa = self_attention and (i == len(sz_chg_idxs) - 3)
            unet_block = UBlock(
                up_in_c,
                x_in_c,
                self.sfs[i],
                final_div=not_final,
                blur=do_blur,
                act_cls=act_cls,
                init=init,
                norm_type=norm_type,
                **kwargs
            ).eval()
            layers.append(unet_block)
            x = unet_block(x)

        ni = x.shape[1]
        if imsize != sizes[0][-2:]:
            layers.append(PixelShuffle_ICNR(ni, act_cls=act_cls, norm_type=norm_type))
        layers.append(ResizeToOrig())
        if last_cross:
            layers.append(MergeLayer(dense=True))
            ni += 3
            layers.append(
                ResBlock(
                    1,
                    ni,
                    ni // 2 if bottle else ni,
                    act_cls=act_cls,
                    norm_type=norm_type,
                    **kwargs
                )
            )
        layers += [
            ConvLayer(ni, n_out, ks=1, act_cls=None, norm_type=norm_type, **kwargs)
        ]
        apply_init(nn.Sequential(layers[3], layers[-2]), init)
        # apply_init(nn.Sequential(layers[2]), init)
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        super().__init__(*layers)

    def __del__(self):
        if hasattr(self, "sfs"):
            self.sfs.remove()


def create_dynamic_unet():
    net = create_body(resnet18())
    net = DynamicUnet(net, 1, (224, 224))
    return net
