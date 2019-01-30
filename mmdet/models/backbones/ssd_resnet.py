# -*- coding: utf-8 -*-
# adjust by @laycoding
import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint

from mmdet.ops import DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_norm_layer

@BACKBONES.register_module
class SSDResNet(ResNet):
    """ResNet backbone for SSD series.

    Args:
        input_size (int): unlike two-stage detector, ssd series use fixed size of input image, from {300, 512}
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        l2_norm_scale (float): Used to norm the feats from different level
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }
    # extra conv layers setting
    extra_setting = {
        300: (256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256),
        512: (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    }
    def __init__(self, input_size, l2_norm_scale=20., **kwargs):
        super(SSDResNet, self).__init__(**kwargs)
        assert input_size in (300, 512)
        self.input_size = input_size
        self.extra = self._make_extra_layers(self.extra_setting[input_size])
        #NB: just norm fist out stage as the paper did
        for name, module in self.named_children():
            if name.endswith("layer"+str(out_indices[0]+1)):
                norm_channel_dim = module[-1].conv3.out_channels
        self.l2_norm = L2Norm(norm_channel_dim, l2_norm_scale)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m, 'conv2_offset'):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

        for m in self.extra.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

        constant_init(self.l2_norm, self.l2_norm.scale)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        for i, layer in enumerate(self.extra):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                outs.append(x)
        #norm the first stage
        outs[0] = self.l2_norm(outs[0])
        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def _make_extra_layers(self, outplanes):
        layers = []
        kernel_sizes = (1, 3)
        num_layers = 0
        outplane = None
        for i in range(len(outplanes)):
            if self.inplanes == 'S':
                self.inplanes = outplane
                continue
            k = kernel_sizes[num_layers % 2]
            if outplanes[i] == 'S':
                outplane = outplanes[i + 1]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=2, padding=1)
            else:
                outplane = outplanes[i]
                conv = nn.Conv2d(
                    self.inplanes, outplane, k, stride=1, padding=0)
            layers.append(conv)
            self.inplanes = outplanes[i]
            num_layers += 1
        if self.input_size == 512:
            layers.append(nn.Conv2d(self.inplanes, 256, 4, padding=1))

        return nn.Sequential(*layers)

class L2Norm(nn.Module):

    def __init__(self, n_dims, scale=20., eps=1e-10):
        super(L2Norm, self).__init__()
        self.n_dims = n_dims
        self.weight = nn.Parameter(torch.Tensor(self.n_dims))
        self.eps = eps
        self.scale = scale

    def forward(self, x):
        norm = x.pow(2).sum(1, keepdim=True).sqrt() + self.eps
        return self.weight[None, :, None, None].expand_as(x) * x / norm
