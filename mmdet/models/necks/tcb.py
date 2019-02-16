import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, kaiming_init, constant_init

from ..utils import ConvModule, build_norm_layer
from ..registry import NECKS
from ..backbones import BasicBlock
from ..backbones.resnet import make_res_layer, conv3x3
from .fpn import FPN


@NECKS.register_module
class TCB(FPN):
    '''transform block in refinedet, 
            conv
            relu
            conv
            ↓
            + ← upsample
            ↓
            relu
            conv
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 start_level=0,
                 end_level=-1,
                 num_outs=4,
                 **kwargs):
        super(TCB, self).__init__(in_channels, out_channels, num_outs, **kwargs)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        if end_level == -1:
            self.end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            # TCB block
            l_conv = make_res_layer(TCBBlock,
               in_channels[i],
               out_channels,
               1,
               stride=1,
               dilation=1,
               style='pytorch',
               normalize=dict(type='BN'))
            #NB: diff from the origin paper,we use simple 3x3 conv instead of residual block
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                normalize=self.normalize,
                bias=self.with_bias,
                activation=self.activation,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
                # kaiming_init(m, nonlinearity="leak_relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        return tuple(outs)

class TCBBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN'),
                 dcn=None):
        super(TCBBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out