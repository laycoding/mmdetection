# # -*- coding: utf-8 -*-
# Atlab@laycoding
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import (AnchorGenerator, anchor_target, refined_anchor_target, weighted_smoothl1,
                        multi_apply)
from .anchor_head import AnchorHead
from ..registry import HEADS


@HEADS.register_module
class RefineDetHead(AnchorHead):

    def __init__(self,
                 input_size=300,
                 num_classes=81,
                 in_channels=([512, 256], [1024, 256], [2048, 256], [512, 256]),
                 anchor_strides=(8, 16, 32, 64),
                 basesize_ratio_range=(0.1, 0.9),
                 anchor_ratios=([2], [2, 3], [2, 3], [2]),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 objectness_score=0.01):
        super(AnchorHead, self).__init__()
        self.objectness_score = objectness_score
        self.input_size = input_size
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.cls_out_channels = num_classes
        num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
        arm_reg_convs = []
        arm_cls_convs = []
        odm_reg_convs = []
        odm_cls_convs = []
        for i in range(len(in_channels)):
            arm_reg_convs.append(
                nn.Conv2d(
                    in_channels[i][0],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            arm_cls_convs.append(
                nn.Conv2d(
                    in_channels[i][0],
                    num_anchors[i] * 2,
                    kernel_size=3,
                    padding=1))
            odm_reg_convs.append(
                nn.Conv2d(
                    in_channels[i][1],
                    num_anchors[i] * 4,
                    kernel_size=3,
                    padding=1))
            odm_cls_convs.append(
                nn.Conv2d(
                    in_channels[i][1],
                    num_anchors[i] * num_classes,
                    kernel_size=3,
                    padding=1))
        self.arm_reg_convs = nn.ModuleList(arm_reg_convs)
        self.arm_cls_convs = nn.ModuleList(arm_cls_convs)
        self.odm_reg_convs = nn.ModuleList(odm_reg_convs)
        self.odm_cls_convs = nn.ModuleList(odm_cls_convs)

        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        # To be clarify, the original implementation is also intricate
        step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
        min_sizes = []
        max_sizes = []
        for r in range(int(min_ratio), int(max_ratio) + 1, step):
            min_sizes.append(int(input_size * r / 100))
            max_sizes.append(int(input_size * (r + step) / 100))
        if input_size == 300:
            assert basesize_ratio_range[0] in [0.15, 0.2]
            if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
            elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
                min_sizes.insert(0, int(input_size * 10 / 100))
                max_sizes.insert(0, int(input_size * 20 / 100))
        elif input_size == 512:
            assert basesize_ratio_range[0] in [0.1, 0.15]
            if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
                min_sizes.insert(0, int(input_size * 4 / 100))
                max_sizes.insert(0, int(input_size * 10 / 100))
            elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
                min_sizes.insert(0, int(input_size * 7 / 100))
                max_sizes.insert(0, int(input_size * 15 / 100))
        # real sizes of anchors
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
            # 1 1/r r '√scale' as the @weiliu implemented
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
        self.use_focal_loss = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform', bias=0)

    def forward(self, feats, refined_feats):
        # arm module
        arm_cls_scores = []
        arm_bbox_preds = []
        for arm_feat, arm_reg_conv, arm_cls_conv in zip(feats, self.arm_reg_convs,
                                            self.arm_cls_convs):
            arm_cls_scores.append(arm_cls_conv(arm_feat))
            arm_bbox_preds.append(arm_reg_conv(arm_feat))

        # odm module
        odm_cls_scores = []
        odm_bbox_preds = []
        for odm_feat, odm_reg_conv, odm_cls_conv in zip(refined_feats, self.odm_reg_convs,
                                            self.odm_cls_convs):
            odm_cls_scores.append(odm_cls_conv(odm_feat))
            odm_bbox_preds.append(odm_reg_conv(odm_feat))

        return arm_cls_scores, arm_bbox_preds, odm_cls_scores, odm_bbox_preds

    def multiboxloss_single(self, cls_score, bbox_pred, labels, label_weights,
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

        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls[None], loss_reg

    def multiboxloss(self, cls_scores, bbox_preds, gt_bboxes, gt_labels, img_metas,
             cfg, use_arm=False, arm_cls_scores=None, arm_bbox_preds=None):
        if arm_bbox_preds is None or arm_cls_scores is None:
            assert use_arm==False
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        if not use_arm:
            cls_reg_targets = anchor_target(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                self.target_means,
                self.target_stds,
                cfg,
                gt_labels_list=gt_labels,
                label_channels=1,
                sampling=False, 
                unmap_outputs=False)
        else:
            #arrange the prediction
            num_images = len(img_metas)
            arm_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, 2) for s in arm_cls_scores
            ], 1)
            # [num_imgs, preds, num_classes]
            arm_bbox_preds = torch.cat([
                b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
                for b in arm_bbox_preds
            ], -2)
            # [num_imgs, preds, 4]
            # get the refined anchors, filter and assign them to gt using arm prediction
            cls_reg_targets = refined_anchor_target(
                anchor_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                self.target_means,
                self.target_stds,
                cfg,
                gt_labels_list=gt_labels,
                label_channels=1,
                sampling=False,
                unmap_outputs=False,
                arm_cls_scores=arm_cls_scores,
                arm_bbox_preds=arm_bbox_preds)          
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_images = len(img_metas)
        if not use_arm:
            all_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, 2) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)
            all_labels[all_labels>0] = 1
        else:
            all_cls_scores = torch.cat([
                s.permute(0, 2, 3, 1).reshape(
                    num_images, -1, self.cls_out_channels) for s in cls_scores
            ], 1)
            all_labels = torch.cat(labels_list, -1).view(num_images, -1)

        all_label_weights = torch.cat(label_weights_list, -1).view(
            num_images, -1)
        all_bbox_preds = torch.cat([
            b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
            for b in bbox_preds
        ], -2)
        all_bbox_targets = torch.cat(bbox_targets_list, -2).view(
            num_images, -1, 4)
        all_bbox_weights = torch.cat(bbox_weights_list, -2).view(
            num_images, -1, 4)

        losses_cls, losses_reg = multi_apply(
            self.multiboxloss_single,
            all_cls_scores,
            all_bbox_preds,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            num_total_samples=num_total_pos,
            cfg=cfg)
        return dict(loss_cls=losses_cls, loss_reg=losses_reg)

    def loss(self, arm_cls_scores, arm_bbox_preds, odm_cls_scores, odm_bbox_preds,
             gt_bboxes, gt_labels, img_metas, cfg):
        arm_losses = self.multiboxloss(arm_cls_scores, arm_bbox_preds,
                                         gt_bboxes, gt_labels, img_metas, cfg)
        odm_losses = self.multiboxloss(odm_cls_scores, odm_bbox_preds,
                                         gt_bboxes, gt_labels, img_metas, cfg,
                                         use_arm=True, arm_cls_scores=arm_cls_scores,
                                         arm_bbox_preds=arm_bbox_preds)

        return dict(arm_loss_cls=arm_losses['loss_cls'],
                     arm_loss_reg=arm_losses['loss_reg'],
                     odm_loss_cls=odm_losses['loss_cls'],
                     odm_loss_reg=odm_losses['loss_reg'])
    # detection out
    def get_bboxes(self, arm_cls_scores, arm_bbox_preds, odm_cls_scores, odm_bbox_preds,
                 img_metas, cfg, rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(cls_scores[i].size()[-2:],
                                                   self.anchor_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            arm_cls_score_list = [
                arm_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            arm_bbox_pred_list = [
                arm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            odm_cls_score_list = [
                odm_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            odm_bbox_pred_list = [
                odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(arm_cls_score_list, arm_bbox_pred_list,
                                               odm_cls_score_list, odm_bbox_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          arm_cls_scores,
                          arm_bbox_preds,
                          odm_cls_scores,
                          odm_bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(mlvl_anchors) == len(arm_bbox_preds) == len(arm_cls_scores)
                                 == len(odm_cls_scores) == len(odm_bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for arm_cls_score, arm_bbox_pred, odmm_cls_score, odm_bbox_pred, anchors in zip(arm_cls_scores,
                arm_bbox_preds, odm_cls_scores, odm_bbox_preds, mlvl_anchors):
            # NB: do this postprocess in every single img
            assert arm_cls_score.size()[-2:] == arm_bbox_pred.size()[-2:]
            assert odm_cls_score.size()[-2:] == odm_bbox_pred.size()[-2:]
            arm_cls_score = arm_cls_score.permute(1, 2, 0).reshape(
                -1, 2)
            odm_cls_score = odm_cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)

            assert self.use_sigmoid_cls is False
            arm_scores = arm_cls_score.softmax(-1)
            odm_scores = odm_cls_score.softmax(-1)
            arm_bbox_pred = arm_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            odm_bbox_pred = odm_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter anchors by arm objectness score
            arm_max_scores, _ = arm_scores[:, 1:].max(dim=1)
            odm_scores[arm_max_scores<=self.objectness_score] = 0
            # decode the arm box pred, TODO@laycoding prefilter before decode the arm pred
            # NB: delta2bbox will clamp the anchor inside the pic which may affect the performance
            arm_bboxes = delta2bbox(anchors, arm_bbox_pred, self.target_means,
                    self.target_stds, img_shape)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and odm_scores.shape[0] > nms_pre:
                max_scores, _ = odm_scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                arm_bboxes = arm_bboxes[topk_inds, :]
                odm_bbox_pred = odm_bbox_pred[topk_inds, :]
                odm_scores = odm_scores[topk_inds, :]
            bboxes = delta2bbox(arm_bboxes, odm_bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)

        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms, cfg.max_per_img)
        return det_bboxes, det_labels
