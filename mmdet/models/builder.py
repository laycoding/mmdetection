from mmcv.runner import obj_from_dict
from torch import nn

from . import (backbones, necks, roi_extractors, anchor_heads, bbox_heads,
               mask_heads)


def _build_module(cfg, parrent=None, default_args=None):
    return cfg if isinstance(cfg, nn.Module) else obj_from_dict(
        cfg, parrent, default_args)


def build(cfg, parrent=None, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent, default_args)


def build_backbone(cfg):
    return build(cfg, backbones)


def build_neck(cfg):
    return build(cfg, necks)


def build_anchor_head(cfg):
    return build(cfg, anchor_heads)


def build_roi_extractor(cfg):
    return build(cfg, roi_extractors)


def build_bbox_head(cfg):
    return build(cfg, bbox_heads)


def build_mask_head(cfg):
    return build(cfg, mask_heads)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    from . import detectors
    return build(cfg, detectors, dict(train_cfg=train_cfg, test_cfg=test_cfg))
