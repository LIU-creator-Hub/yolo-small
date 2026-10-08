"""SAD-Loss: Shape-Aware Dynamic detection loss for steel surface defects.

It extends the standard YOLO detection loss (CIoU + DFL) with two new terms
that are tailored to NEU-DET:

  1) shape / aspect-ratio consistency  -> helps elongated defects (crazing, scratches)
  2) Normalized Wasserstein Distance   -> helps small and thin defects whose IoU is
                                          very sensitive to a few pixels

Only the bounding-box branch is modified; classification and DFL stay exactly as
in ultralytics, and both new terms are multiplied by small gains so training
remains stable (set the gains to 0.0 to recover the original loss).
"""

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class SADBboxLoss(BboxLoss):
    """CIoU + DFL + shape-ratio + NWD bounding-box loss."""

    def __init__(self, reg_max=16, shape_gain=0.05, nwd_gain=0.05, nwd_c=6.0):
        super().__init__(reg_max)
        self.shape_gain = float(shape_gain)
        self.nwd_gain = float(nwd_gain)
        self.nwd_c = float(nwd_c)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb, tb = pred_bboxes[fg_mask], target_bboxes[fg_mask]

        # --- standard CIoU ---
        iou = bbox_iou(pb, tb, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # --- standard DFL ---
        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.zeros((), device=pred_dist.device, dtype=pred_dist.dtype)

        # --- new term 1: aspect-ratio consistency (elongated defects) ---
        pw = (pb[:, 2] - pb[:, 0]).clamp(min=1e-6)
        ph = (pb[:, 3] - pb[:, 1]).clamp(min=1e-6)
        tw = (tb[:, 2] - tb[:, 0]).clamp(min=1e-6)
        th = (tb[:, 3] - tb[:, 1]).clamp(min=1e-6)
        loss_shape = ((torch.log(pw / ph) - torch.log(tw / th)).abs() * weight).sum() / target_scores_sum

        # --- new term 2: normalized Wasserstein distance (small / thin defects) ---
        pcx, pcy = (pb[:, 0] + pb[:, 2]) * 0.5, (pb[:, 1] + pb[:, 3]) * 0.5
        tcx, tcy = (tb[:, 0] + tb[:, 2]) * 0.5, (tb[:, 1] + tb[:, 3]) * 0.5
        d2 = (
            (pcx - tcx) ** 2
            + (pcy - tcy) ** 2
            + (pw * 0.5 - tw * 0.5) ** 2
            + (ph * 0.5 - th * 0.5) ** 2
        )
        nwd = torch.exp(-(d2 / self.nwd_c).clamp(max=30.0))
        loss_nwd = ((1.0 - nwd) * weight).sum() / target_scores_sum

        return loss_iou + self.shape_gain * loss_shape + self.nwd_gain * loss_nwd, loss_dfl


class SADDetectionLoss(v8DetectionLoss):
    """YOLO detection loss with the shape-aware bounding-box branch."""

    def __init__(self, model, tal_topk=10, shape_gain=0.05, nwd_gain=0.05, nwd_c=6.0):
        super().__init__(model, tal_topk)
        self.bbox_loss = SADBboxLoss(self.reg_max, shape_gain, nwd_gain, nwd_c).to(self.device)


class SADTrainer(DetectionTrainer):
    """DetectionTrainer that installs SADDetectionLoss on the model."""

    shape_gain = 0.05
    nwd_gain = 0.05
    nwd_c = 6.0

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.criterion = SADDetectionLoss(
            model, shape_gain=self.shape_gain, nwd_gain=self.nwd_gain, nwd_c=self.nwd_c
        )
        return model
