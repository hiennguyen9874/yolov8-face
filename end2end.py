import random

import torch
import torch.nn as nn


class ORT_NMS(torch.autograd.Function):
    """ONNX-Runtime NMS operation"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,), device=device).sort()[0]
        idxs = torch.arange(100, 100 + num_det, device=device)
        zeros = torch.zeros((num_det,), dtype=torch.int64, device=device)
        selected_indices = torch.cat(
            [batches[None], zeros[None], idxs[None]], 0
        ).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        max_output_boxes_per_class=torch.tensor([100]),
        iou_threshold=torch.tensor([0.45]),
        score_threshold=torch.tensor([0.25]),
    ):
        return g.op(
            "NonMaxSuppression",
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
            center_point_box_i=0,
        )


class ONNX_ORT(nn.Module):
    """onnx module with ONNX-Runtime NMS operation."""

    def __init__(
        self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None, max_wh=640
    ):
        super().__init__()
        self.max_wh = max_wh
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.convert_matrix = torch.tensor(
            [
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [-0.5, 0, 0.5, 0],
                [0, -0.5, 0, 0.5],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def forward(self, x):
        boxes = x[:, :, :4]
        scores = x[:, :, 4:5]
        lmks = x[:, :, [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]]
        lmks_mask = x[:, :, [7, 10, 13, 16, 19]]

        # convert cxcywh => xyxy
        boxes @= self.convert_matrix
        max_score, category_id = scores.max(2, keepdim=True)

        # Per class nms
        dis = category_id.float() * self.max_wh
        nmsbox = boxes + dis

        # convert to [num_batches, num_classes, spatial_dimension] format
        max_score_tp = max_score.transpose(1, 2).contiguous()

        selected_indices = ORT_NMS.apply(
            nmsbox,
            max_score_tp,
            self.max_obj,
            self.iou_threshold,
            self.score_threshold,
        )
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        selected_boxes = boxes[X, Y, :]
        selected_categories = category_id[X, Y, :].float()
        selected_scores = max_score[X, Y, :]
        selected_lmks = lmks[X, Y, :]
        selected_lmks_mask = lmks_mask[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.cat(
            [
                X,
                selected_boxes,
                selected_categories,
                selected_scores,
                selected_lmks,
                selected_lmks_mask,
            ],
            1,
        )


class TRT_NMS(torch.autograd.Function):
    """TensorRT NMS operation using EfficientNMSCustom_TRT"""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        device = boxes.device
        dtype = boxes.dtype

        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(
            0, max_output_boxes, (batch_size, 1), device=device, dtype=torch.int32
        )
        det_boxes = torch.randn(
            batch_size, max_output_boxes, 4, device=device, dtype=dtype
        )
        det_scores = torch.randn(
            batch_size, max_output_boxes, device=device, dtype=dtype
        )
        det_classes = torch.randint(
            0,
            num_classes,
            (batch_size, max_output_boxes),
            device=device,
            dtype=torch.int32,
        )
        det_indices = torch.randint(
            0,
            num_boxes,
            (batch_size, max_output_boxes),
            device=device,
            dtype=torch.int32,
        )
        return num_det, det_boxes, det_scores, det_classes, det_indices

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        background_class=-1,
        iou_threshold=0.45,
        max_output_boxes=100,
        score_activation=0,
        score_threshold=0.25,
        box_coding=1,
    ):
        out = g.op(
            "TRT::EfficientNMSCustom_TRT",
            boxes,
            scores,
            background_class_i=background_class,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,
            score_threshold_f=score_threshold,
            outputs=5,
        )
        num_det, det_boxes, det_scores, det_classes, det_indices = out
        return num_det, det_boxes, det_scores, det_classes, det_indices


class ONNX_TRT(nn.Module):
    """onnx module with TensorRT NMS operation."""

    def __init__(
        self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None
    ):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.background_class = (-1,)
        self.box_coding = (1,)
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = "1"
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        boxes = x[:, :, :4]
        scores = x[:, :, 4:5]
        lmks = x[:, :, [5, 6, 8, 9, 11, 12, 14, 15, 17, 18]]
        lmks_mask = x[:, :, [7, 10, 13, 16, 19]]

        batch_size, _, _ = scores.shape
        total_object = batch_size * self.max_obj

        num_det, det_boxes, det_scores, det_classes, det_indices = TRT_NMS.apply(
            boxes,
            scores,
            self.background_class,
            self.iou_threshold,
            self.max_obj,
            self.score_activation,
            self.score_threshold,
        )
        batch_indices = torch.ones_like(det_indices) * torch.arange(
            batch_size, device=self.device, dtype=torch.int32
        ).unsqueeze(1)
        batch_indices = batch_indices.view(total_object).to(torch.long)
        det_indices = det_indices.view(total_object).to(torch.long)

        det_lmks = lmks[batch_indices, det_indices].view(batch_size, self.max_obj, 10)
        det_lmks_mask = lmks_mask[batch_indices, det_indices].view(
            batch_size, self.max_obj, 5
        )

        return num_det, det_boxes, det_scores, det_classes, det_lmks, det_lmks_mask


class End2End(nn.Module):
    """export onnx or tensorrt model with NMS operation."""

    def __init__(
        self,
        model,
        max_obj=100,
        iou_thres=0.45,
        score_thres=0.25,
        max_wh=None,
        trt=False,
        device=None,
    ):
        super().__init__()
        device = device if device is not None else torch.device("cpu")
        assert isinstance(max_wh, (int)) or max_wh is None
        self.model = model.to(device)
        self.patch_model = ONNX_TRT if trt else ONNX_ORT
        self.end2end = self.patch_model(
            max_obj=max_obj,
            iou_thres=iou_thres,
            score_thres=score_thres,
            max_wh=max_wh,
            device=device,
        )
        self.end2end.eval()
        self.end2end = self.end2end.to(device)

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x
