import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init, xavier_init
from mmdet.core import (multi_apply, multiclass_nms, distance2bbox,
                        weighted_sigmoid_focal_loss, select_iou_loss)
from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox, weighted_smoothl1, 
                        multi_apply)
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule


@HEADS.register_module
class HybridAFBHead(nn.Module):
    """Feature Selective Hybrid Anchor-Free and Anchor-Based Head

    Args:
    anchor free part:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        stacked_convs (int): Number of conv layers before head.
        norm_factor (float): Distance normalization factor.
        feat_strides (Iterable): Feature strides.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    anchor based part:(adjust from ssd head)
        input_size=300,
        num_classes=21,
        in_channels=(512, 1024, 512, 256, 256, 256),
        anchor_strides=(8, 16, 32, 64, 100, 300),
        basesize_ratio_range=(0.1, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(1.0, 1.0, 1.0, 1.0)):
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 norm_factor=4.0,
                 feat_strides=[8, 16, 32, 64, 128],
                 conv_cfg=None,
                 norm_cfg=None,
                 input_size=300,
                 ab_in_channels=(512, 1024, 512, 256, 256),
                 anchor_strides=(8, 16, 32, 64, 100),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(HybridAFBHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.norm_factor = norm_factor
        self.feat_strides = feat_strides
        self.cls_out_channels_af = self.num_classes - 1
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        # anchor based part
        self.input_size = input_size
        self.num_classes = num_classes
        self.ab_in_channels = ab_in_channels
        self.cls_out_channels_ab = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        ab_reg_convs = []
        ab_cls_convs = []
        for i in range(len(ab_in_channels)):
            ab_reg_convs.append(
                nn.Conv2d(
                    ab_in_channels[i],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            ab_cls_convs.append(
                nn.Conv2d(
                    ab_in_channels[i],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.ab_reg_convs = nn.ModuleList(ab_reg_convs)
        self.ab_cls_convs = nn.ModuleList(ab_cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (len(ab_in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        self.anchor_generators = []
        self.anchor_strides = anchor_strides
        for k in range(len(anchor_strides)):
            base_size = min_sizes[k]
            stride = anchor_strides[k]
            ctr = ((stride - 1) / 2., (stride - 1) / 2.)
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            ratios = [1.]
            for r in anchor_ratios[k]:
                ratios += [1 / r, r]  # 4 or 6 ratio
            anchor_generator = AnchorGenerator(
                base_size, scales, ratios, scale_major=False, ctr=ctr)
            indices = list(range(len(ratios)))
            indices.insert(1, len(indices))
            anchor_generator.base_anchors = torch.index_select(
                anchor_generator.base_anchors, 0, torch.LongTensor(indices))
            self.anchor_generators.append(anchor_generator)

        self.target_means = target_means
        self.target_stds = target_stds
        self.use_sigmoid_cls = False
        self.cls_focal_loss = False

        self._init_layers()

    def _init_layers(self):
        # todo@laycoding: mv the anchor based layers initialization here
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.fsaf_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels_af, 3, padding=1)
        self.fsaf_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fsaf_cls, std=0.01, bias=bias_cls)
        normal_init(self.fsaf_reg, std=0.01, bias=0.1)
        # the anchor based part
        for m in self.ab_cls_convs:
            xavier_init(m, distribution='uniform', bias=0)
        for m in self.ab_reg_convs:
            xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats):
        ab_cls_scores = []
        ab_bbox_preds = []
        af_cls_scores = []
        af_bbox_preds = []
        stacked_cls_feats = feats
        stacked_reg_feats = feats
        cls_feats = []
        reg_feats = []
        # head of the head 233
        for cls_feat in stacked_cls_feats:
            for cls_conv in self.cls_convs:
                cls_feat = cls_conv(cls_feat)
            cls_feats.append(cls_feat)

        for reg_feat in stacked_reg_feats:
            for reg_conv in self.reg_convs:
                reg_feat = reg_conv(reg_feat)
            reg_feats.append(reg_feat)   
        # anchor based part
        for cls_feat, reg_feat, reg_conv, cls_conv in zip(cls_feats, reg_feats, self.ab_reg_convs,
                                            self.ab_cls_convs):
            ab_cls_scores.append(cls_conv(cls_feat))
            ab_bbox_preds.append(reg_conv(reg_feat))
        # anchor free part
        for cls_feat, reg_feat in zip(cls_feats, reg_feats):
            af_cls_scores.append(self.fsaf_cls(cls_feat))
            af_bbox_preds.append(self.relu(self.fsaf_reg(reg_feat)))

        return ab_cls_scores, ab_bbox_preds, af_cls_scores, af_bbox_preds

    def af_loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_locs, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels_af)
        loss_cls = weighted_sigmoid_focal_loss(
            cls_score,
            labels,
            label_weights,
            cfg.gamma,
            cfg.alpha,
            avg_factor=num_total_samples)
        # localization loss
        if bbox_targets.size(0) == 0:
            loss_reg = bbox_pred.new_zeros(1)
        else:
            bbox_pred = bbox_pred.permute(0, 2, 3, 1)
            bbox_pred = bbox_pred[bbox_locs[:, 0], bbox_locs[:, 1],
                                  bbox_locs[:, 2], :]
            loss_reg = select_iou_loss(
                bbox_pred,
                bbox_targets,
                cfg.bbox_reg_weight,
                avg_factor=num_total_samples)
        return loss_cls, loss_reg

    def ab_loss_single(self, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        loss_cls_all = F.cross_entropy(
            cls_score, labels, reduction='none') * label_weights
        pos_inds = (labels > 0).nonzero().view(-1)
        neg_inds = (labels == 0).nonzero().view(-1)

        num_pos_samples = pos_inds.size(0)
        num_neg_samples = cfg.neg_pos_ratio * num_pos_samples
        if num_neg_samples > neg_inds.size(0):
            num_neg_samples = neg_inds.size(0)
        topk_loss_cls_neg, _ = loss_cls_all[neg_inds].topk(num_neg_samples)
        loss_cls_pos = loss_cls_all[pos_inds].sum()
        loss_cls_neg = topk_loss_cls_neg.sum()
        loss_cls = (loss_cls_pos + loss_cls_neg) / num_total_samples

        loss_bbox = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_bbox

    def loss(self,
             ab_cls_scores,
             ab_bbox_preds,
             af_cls_scores,
             af_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # anchor free part
        af_cls_reg_targets = self.point_target(
            af_cls_scores,
            af_bbox_preds,
            gt_bboxes,
            img_metas,
            cfg,
            gt_labels_list=gt_labels,
            gt_bboxes_ignore_list=gt_bboxes_ignore)
        # if cls_reg_targets is None:
        #     return None
        (af_labels_list, af_label_weights_list, af_bbox_targets_list, af_bbox_locs_list,
         af_num_total_pos, af_num_total_neg) = af_cls_reg_targets
        af_num_total_samples = af_num_total_pos
        af_losses_cls, af_losses_reg = multi_apply(
            self.af_loss_single,
            af_cls_scores,
            af_bbox_preds,
            af_labels_list,
            af_label_weights_list,
            af_bbox_targets_list,
            af_bbox_locs_list,
            num_total_samples=af_num_total_samples,
            cfg=cfg)
        # anchor based part
        featmap_sizes = [featmap.size()[-2:] for featmap in ab_cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        ab_cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=1,
            sampling=False,
            unmap_outputs=False)
        if ab_cls_reg_targets is None:
            return None
        (ab_labels_list, ab_label_weights_list, ab_bbox_targets_list, ab_bbox_weights_list,
         ab_num_total_pos, ab_num_total_neg) = ab_cls_reg_targets

        num_images = len(img_metas)
        ab_all_cls_scores = torch.cat([
            s.permute(0, 2, 3, 1).reshape(
                num_images, -1, self.cls_out_channels_ab) for s in ab_cls_scores
        ], 1)
        ab_all_labels = torch.cat(ab_labels_list, -1).view(num_images, -1)
        ab_all_label_weights = torch.cat(ab_label_weights_list,
                                      -1).view(num_images, -1)
        ab_all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in ab_bbox_preds
        ], -2)
        ab_all_bbox_targets = torch.cat(ab_bbox_targets_list,
                                     -2).view(num_images, -1, 4)
        ab_all_bbox_weights = torch.cat(ab_bbox_weights_list,
                                     -2).view(num_images, -1, 4)

        ab_losses_cls, ab_losses_bbox = multi_apply(
            self.ab_loss_single,
            ab_all_cls_scores,
            ab_all_bbox_preds,
            ab_all_labels,
            ab_all_label_weights,
            ab_all_bbox_targets,
            ab_all_bbox_weights,
            num_total_samples=ab_num_total_pos,
            cfg=cfg)
        return dict(ab_loss_cls=ab_losses_cls, ab_loss_bbox=ab_losses_bbox, af_loss_cls=af_losses_cls, af_loss_reg=af_losses_reg)

    def point_target(self,
                     cls_scores,
                     bbox_preds,
                     gt_bboxes,
                     img_metas,
                     cfg,
                     gt_labels_list=None,
                     gt_bboxes_ignore_list=None):
        num_imgs = len(img_metas)
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        # split net outputs w.r.t. images
        num_levels = len(self.feat_strides)
        assert len(cls_scores) == len(bbox_preds) == num_levels
        cls_score_list = []
        bbox_pred_list = []
        for img_id in range(num_imgs):
            cls_score_list.append(
                [cls_scores[i][img_id].detach() for i in range(num_levels)])
            bbox_pred_list.append(
                [bbox_preds[i][img_id].detach() for i in range(num_levels)])

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_locs,
         num_pos_list, num_neg_list) = multi_apply(
             self.point_target_single,
             cls_score_list,
             bbox_pred_list,
             gt_bboxes,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             cfg=cfg)
        # correct image index in bbox_locs
        for i in range(num_imgs):
            for lvl in range(num_levels):
                all_bbox_locs[i][lvl][:, 0] = i

        # sampled points of all images
        num_total_pos = sum([max(num, 1) for num in num_pos_list])
        num_total_neg = sum([max(num, 1) for num in num_neg_list])
        # combine targets to a list w.r.t. multiple levels
        labels_list = self.images_to_levels(all_labels, num_imgs, num_levels,
                                            True)
        label_weights_list = self.images_to_levels(all_label_weights, num_imgs,
                                                   num_levels, True)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_imgs,
                                                  num_levels, False)
        bbox_locs_list = self.images_to_levels(all_bbox_locs, num_imgs,
                                               num_levels, False)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_locs_list, num_total_pos, num_total_neg)

    def point_target_single(self, cls_score_list, bbox_pred_list, gt_bboxes,
                            gt_bboxes_ignore, gt_labels, img_meta, cfg):
        num_levels = len(self.feat_strides)
        assert len(cls_score_list) == len(bbox_pred_list) == num_levels
        feat_lvls = self.feat_level_select(cls_score_list, bbox_pred_list,
                                           gt_bboxes, gt_labels, cfg)
        labels = []
        label_weights = []
        bbox_targets = []
        bbox_locs = []
        device = bbox_pred_list[0].device
        img_h, img_w, _ = img_meta['pad_shape']
        for lvl in range(num_levels):
            stride = self.feat_strides[lvl]
            norm = stride * self.norm_factor
            inds = torch.nonzero(feat_lvls == lvl).squeeze(-1)
            h, w = cls_score_list[lvl].size()[-2:]
            valid_h = min(int(np.ceil(img_h / stride)), h)
            valid_w = min(int(np.ceil(img_w / stride)), w)

            _labels = torch.zeros_like(
                cls_score_list[lvl][0], dtype=torch.long)
            _label_weights = torch.zeros_like(
                cls_score_list[lvl][0], dtype=torch.float)
            _label_weights[:valid_h, :valid_w] = 1.
            _bbox_targets = bbox_pred_list[lvl].new_zeros((0, 4),
                                                          dtype=torch.float)
            _bbox_locs = bbox_pred_list[lvl].new_zeros((0, 3),
                                                       dtype=torch.long)

            if len(inds) > 0:
                boxes = gt_bboxes[inds, :]
                classes = gt_labels[inds]
                proj_boxes = boxes / stride
                ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                    proj_boxes, cfg.ignore_scale, w, h)
                pos_x1, pos_y1, pos_x2, pos_y2 = self.prop_box_bounds(
                    proj_boxes, cfg.pos_scale, w, h)
                for i in range(len(inds)):
                    # setup classification ground-truth
                    _labels[pos_y1[i]:pos_y2[i], pos_x1[i]:
                            pos_x2[i]] = classes[i]
                    _label_weights[ig_y1[i]:ig_y2[i], ig_x1[i]:ig_x2[i]] = 0.
                    _label_weights[pos_y1[i]:pos_y2[i], pos_x1[i]:
                                   pos_x2[i]] = 1.
                    # setup localization ground-truth
                    locs_x = torch.arange(
                        pos_x1[i], pos_x2[i], device=device, dtype=torch.long)
                    locs_y = torch.arange(
                        pos_y1[i], pos_y2[i], device=device, dtype=torch.long)
                    shift_x = (locs_x.float() + 0.5) * stride
                    shift_y = (locs_y.float() + 0.5) * stride
                    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
                    shifts = torch.stack(
                        (shift_xx, shift_yy, shift_xx, shift_yy), dim=-1)
                    shifts[:, 0] = shifts[:, 0] - boxes[i, 0]
                    shifts[:, 1] = shifts[:, 1] - boxes[i, 1]
                    shifts[:, 2] = boxes[i, 2] - shifts[:, 2]
                    shifts[:, 3] = boxes[i, 3] - shifts[:, 3]
                    _bbox_targets = torch.cat((_bbox_targets, shifts / norm),
                                              dim=0)
                    locs_xx, locs_yy = self._meshgrid(locs_x, locs_y)
                    zeros = torch.zeros_like(locs_xx)
                    locs = torch.stack((zeros, locs_yy, locs_xx), dim=-1)
                    _bbox_locs = torch.cat((_bbox_locs, locs), dim=0)

            labels.append(_labels)
            label_weights.append(_label_weights)
            bbox_targets.append(_bbox_targets)
            bbox_locs.append(_bbox_locs)

        # ignore regions in adjacent pyramids
        for lvl in range(num_levels):
            stride = self.feat_strides[lvl]
            w, h = cls_score_list[lvl].size()[-2:]
            # lower pyramid if exists
            if lvl > 0:
                inds = torch.nonzero(feat_lvls == lvl - 1).squeeze(-1)
                if len(inds) > 0:
                    boxes = gt_bboxes[inds, :]
                    proj_boxes = boxes / stride
                    ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                        proj_boxes, cfg.ignore_scale, w, h)
                    for i in range(len(inds)):
                        label_weights[lvl][ig_y1[i]:ig_y2[i], ig_x1[i]:
                                           ig_x2[i]] = 0.
            # upper pyramid if exists
            if lvl < num_levels - 1:
                inds = torch.nonzero(feat_lvls == lvl + 1).squeeze(-1)
                if len(inds) > 0:
                    boxes = gt_bboxes[inds, :]
                    proj_boxes = boxes / stride
                    ig_x1, ig_y1, ig_x2, ig_y2 = self.prop_box_bounds(
                        proj_boxes, cfg.ignore_scale, w, h)
                    for i in range(len(inds)):
                        label_weights[lvl][ig_y1[i]:ig_y2[i], ig_x1[i]:
                                           ig_x2[i]] = 0.

        # compute number of foreground and background points
        num_pos = 0
        num_neg = 0
        for lvl in range(num_levels):
            npos = bbox_targets[lvl].size(0)
            num_pos += npos
            num_neg += (label_weights[lvl].nonzero().size(0) - npos)
        return (labels, label_weights, bbox_targets, bbox_locs, num_pos,
                num_neg)

    def feat_level_select(self, cls_score_list, bbox_pred_list, gt_bboxes,
                          gt_labels, cfg):
        if cfg.online_select:
            num_levels = len(cls_score_list)
            num_boxes = gt_bboxes.size(0)
            feat_losses = gt_bboxes.new_zeros((num_boxes, num_levels))
            device = bbox_pred_list[0].device
            for lvl in range(num_levels):
                stride = self.feat_strides[lvl]
                norm = stride * self.norm_factor
                cls_score = cls_score_list[lvl].permute(1, 2, 0)  # h x w x C
                bbox_pred = bbox_pred_list[lvl].permute(1, 2, 0)  # h x w x 4
                h, w = cls_score.size()[:2]

                proj_boxes = gt_bboxes / stride
                x1, y1, x2, y2 = self.prop_box_bounds(proj_boxes,
                                                      cfg.pos_scale, w, h)

                for i in range(num_boxes):
                    locs_x = torch.arange(
                        x1[i], x2[i], device=device, dtype=torch.long)
                    locs_y = torch.arange(
                        y1[i], y2[i], device=device, dtype=torch.long)
                    locs_xx, locs_yy = self._meshgrid(locs_x, locs_y)
                    avg_factor = locs_xx.size(0)
                    # classification focal loss
                    scores = cls_score[locs_yy, locs_xx, :]
                    labels = gt_labels[i].repeat(avg_factor)
                    label_weights = torch.ones_like(labels).float()
                    loss_cls = weighted_sigmoid_focal_loss(
                        scores, labels, label_weights, cfg.gamma, cfg.alpha,
                        avg_factor)
                    # localization iou loss
                    deltas = bbox_pred[locs_yy, locs_xx, :]
                    shift_x = (locs_x.float() + 0.5) * stride
                    shift_y = (locs_y.float() + 0.5) * stride
                    shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
                    shifts = torch.stack(
                        (shift_xx, shift_yy, shift_xx, shift_yy), dim=-1)
                    shifts[:, 0] = shifts[:, 0] - gt_bboxes[i, 0]
                    shifts[:, 1] = shifts[:, 1] - gt_bboxes[i, 1]
                    shifts[:, 2] = gt_bboxes[i, 2] - shifts[:, 2]
                    shifts[:, 3] = gt_bboxes[i, 3] - shifts[:, 3]
                    loss_loc = select_iou_loss(deltas, shifts / norm,
                                               cfg.bbox_reg_weight, avg_factor)
                    feat_losses[i, lvl] = loss_cls + loss_loc
            feat_levels = torch.argmin(feat_losses, dim=1)
        else:
            num_levels = len(self.feat_strides)
            lvl0 = cfg.canonical_level
            s0 = cfg.canonical_scale
            assert 0 <= lvl0 < num_levels
            gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0]
            gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1]
            s = torch.sqrt(gt_w * gt_h)
            # FPN Eq. (1)
            feat_levels = torch.floor(lvl0 + torch.log2(s / s0 + 1e-6))
            feat_levels = torch.clamp(feat_levels, 0, num_levels - 1).int()
        return feat_levels

    def xyxy2xcycwh(self, xyxy):
        """Convert [x1 y1 x2 y2] box format to [xc yc w h] format."""
        return torch.cat(
            (0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]),
            dim=1)

    def xcycwh2xyxy(self, xywh):
        """Convert [xc yc w y] box format to [x1 y1 x2 y2] format."""
        return torch.cat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4],
                          xywh[:, 0:2] + 0.5 * xywh[:, 2:4]),
                         dim=1)

    def prop_box_bounds(self, boxes, scale, width, height):
        """Compute proportional box regions.

        Box centers are fixed. Box w and h scaled by scale.
        """
        prop_boxes = self.xyxy2xcycwh(boxes)
        prop_boxes[:, 2:] *= scale
        prop_boxes = self.xcycwh2xyxy(prop_boxes)
        x1 = torch.floor(prop_boxes[:, 0]).clamp(0, width - 1).int()
        y1 = torch.floor(prop_boxes[:, 1]).clamp(0, height - 1).int()
        x2 = torch.ceil(prop_boxes[:, 2]).clamp(1, width).int()
        y2 = torch.ceil(prop_boxes[:, 3]).clamp(1, height).int()
        return x1, y1, x2, y2

    def images_to_levels(self, target, num_imgs, num_levels, is_cls=True):
        level_target = []
        if is_cls:
            for lvl in range(num_levels):
                level_target.append(
                    torch.stack([target[i][lvl] for i in range(num_imgs)],
                                dim=0))
        else:
            for lvl in range(num_levels):
                level_target.append(
                    torch.cat([target[j][lvl] for j in range(num_imgs)],
                              dim=0))
        return level_target

    def generate_points(self,
                        featmap_size,
                        stride=16,
                        device='cuda',
                        dtype=torch.float32):
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device, dtype=dtype) + 0.5
        shift_y = torch.arange(0, feat_h, device=device, dtype=dtype) + 0.5
        shift_x *= stride
        shift_y *= stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        points = torch.stack((shift_xx, shift_yy), dim=-1)
        return points

    def _meshgrid(self, x, y):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        return xx, yy

    def get_bboxes(self, ab_cls_scores, ab_bbox_preds, af_cls_scores, af_bbox_preds, img_metas, cfg,
                   rescale=False):
        # get the bboxes from two method and ensemble them
        num_levels = len(self.feat_strides)
        assert len(ab_cls_scores) == len(ab_bbox_preds) == len(af_cls_scores) == len(af_bbox_preds) == num_levels
        device = af_bbox_preds[0].device
        dtype = af_bbox_preds[0].dtype
        # get the multilevel points based on anchor free outputs
        mlvl_points = [
            self.generate_points(
                af_bbox_preds[i].size()[-2:],
                self.feat_strides[i],
                device=device,
                dtype=dtype) for i in range(num_levels)
        ]
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(ab_cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            af_cls_score_list = [
                af_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            af_bbox_pred_list = [
                af_bbox_preds[i][img_id].detach() * self.feat_strides[i] *
                self.norm_factor for i in range(num_levels)
            ]
            ab_cls_score_list = [
                ab_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            ab_bbox_pred_list = [
                ab_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(af_cls_score_list, af_bbox_pred_list, mlvl_points,
                                               ab_cls_score_list, ab_bbox_pred_list, mlvl_anchors,
                                               img_shape, scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          af_cls_scores,
                          af_bbox_preds,
                          af_mlvl_points,
                          ab_cls_scores,
                          ab_bbox_preds,
                          ab_mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(ab_cls_scores) == len(ab_bbox_preds) == len(af_cls_scores) == len(af_bbox_preds) == len(af_mlvl_points)
        af_mlvl_bboxes = []
        af_mlvl_scores = []
        ab_mlvl_bboxes = []
        ab_mlvl_scores = []
        # the anchor free part
        for af_cls_score, af_bbox_pred, af_points in zip(af_cls_scores, af_bbox_preds,
                                                af_mlvl_points):
            assert af_cls_score.size()[-2:] == af_bbox_pred.size()[-2:]
            af_cls_score = af_cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels_af)
            af_scores = af_cls_score.sigmoid()
            af_bbox_pred = af_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and af_scores.shape[0] > nms_pre:
                max_scores, _ = af_scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                af_bbox_pred = af_bbox_pred[topk_inds, :]
                af_scores = af_scores[topk_inds, :]
                af_points = af_points[topk_inds, :]
            af_bboxes = distance2bbox(af_points, af_bbox_pred, img_shape)
            af_mlvl_bboxes.append(af_bboxes)
            af_mlvl_scores.append(af_scores)
        af_mlvl_bboxes = torch.cat(af_mlvl_bboxes)
        if rescale:
            af_mlvl_bboxes /= af_mlvl_bboxes.new_tensor(scale_factor)
        af_mlvl_scores = torch.cat(af_mlvl_scores)
        af_padding = af_mlvl_scores.new_zeros(af_mlvl_scores.shape[0], 1)
        af_mlvl_scores = torch.cat([af_padding, af_mlvl_scores], dim=1)
        # the anchor based part
        for ab_cls_score, ab_bbox_pred, ab_anchors in zip(ab_cls_scores, ab_bbox_preds,
                                                 ab_mlvl_anchors):
            assert ab_cls_score.size()[-2:] == ab_bbox_pred.size()[-2:]
            ab_cls_score = ab_cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels_ab)
            if self.use_sigmoid_cls:
                ab_scores = ab_cls_score.sigmoid()
            else:
                ab_scores = ab_cls_score.softmax(-1)
            ab_bbox_pred = ab_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and ab_scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = ab_scores.max(dim=1)
                else:
                    max_scores, _ = ab_scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                ab_anchors = ab_anchors[topk_inds, :]
                ab_bbox_pred = ab_bbox_pred[topk_inds, :]
                ab_scores = ab_scores[topk_inds, :]
            ab_bboxes = delta2bbox(ab_anchors, ab_bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            ab_mlvl_bboxes.append(ab_bboxes)
            ab_mlvl_scores.append(ab_scores)
        ab_mlvl_bboxes = torch.cat(ab_mlvl_bboxes)
        if rescale:
            ab_mlvl_bboxes /= ab_mlvl_bboxes.new_tensor(scale_factor)
        ab_mlvl_scores = torch.cat(ab_mlvl_scores)
        if self.use_sigmoid_cls:
            ab_padding = ab_mlvl_scores.new_zeros(ab_mlvl_scores.shape[0], 1)
            ab_mlvl_scores = torch.cat([ab_padding, ab_mlvl_scores], dim=1)
        # ensemble them (N x 4) (N x C)
        mlvl_bboxes = torch.cat((ab_mlvl_bboxes, af_mlvl_bboxes))
        mlvl_scores = torch.cat((ab_mlvl_scores, af_mlvl_scores))
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels

    def get_anchors(self, featmap_sizes, img_metas):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list