import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, kaiming_init, constant_init

# from ..utils import ConvModule, build_norm_layer
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
        self.start_level = start_level

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.end_level):
            # TCB block
            l_conv = TCBBlock(
                 in_channels[i],
                 out_channels,
                 stride=1,
                 dilation=1)
            #NB: diff from the origin paper,we use simple 3x3 conv instead of residual block
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

            # lvl_id = i - self.start_level
            # setattr(self, 'lateral_conv{}'.format(lvl_id), l_conv)
            # setattr(self, 'fpn_conv{}'.format(lvl_id), fpn_conv)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier_init(m, distribution='uniform')
                kaiming_init(m)

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
                laterals[i], scale_factor=2, mode='bilinear', align_corners=False)
        for lateral in laterals:
            lateral = F.relu(lateral, inplace=True)
        # build outputs
        outs = [
            F.relu(self.fpn_convs[i](laterals[i]), inplace=True)
            for i in range(used_backbone_levels)
        ]

        return tuple(outs)

class TCBBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1):
        super(TCBBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, \
            kernel_size=3, stride=stride, padding=dilation)
        self.conv2 = nn.Conv2d(planes, planes, \
            kernel_size=3, stride=stride, padding=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        return out
