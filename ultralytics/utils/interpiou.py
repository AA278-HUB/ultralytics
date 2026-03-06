"""
InterpIoU and D-InterpIoU: Interpolation-based IoU Loss Functions for Object Detection

This module implements InterpIoU and D-InterpIoU, two novel IoU loss functions
that use interpolation techniques to improve bounding box regression in object detection.
"""

import torch


def interpiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    interp_coe: float = 0.98,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate InterpIoU (Interpolation IoU) between bounding boxes.

    InterpIoU uses a fixed interpolation coefficient to create an interpolated bounding box
    between box1 and box2, then computes the IoU between the interpolated box and box2.
    The final IoU is calculated as: IoU + IoU_interp - 1.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
                            Supports shapes like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
                            Supports shapes like (4,), (M, 4), (B, M, 4), or (B, M, 1, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format (center coordinates).
                               If False, input boxes are in (x1, y1, x2, y2) format (corner coordinates).
                               Defaults to True.
        interp_coe (float, optional): Interpolation coefficient in [0, 1]. 
                                     When interp_coe=0, the interpolated box equals box1.
                                     When interp_coe=1, the interpolated box equals box2.
                                     Defaults to 0.98.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): InterpIoU values with the same shape as the broadcast result of box1 and box2.

    Examples:
        >>> box1 = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])  # (N, 4) in xywh format
        >>> box2 = torch.tensor([[12, 12, 20, 20], [32, 32, 40, 40]])  # (M, 4) in xywh format
        >>> iou = interpiou(box1, box2, xywh=True, interp_coe=0.98)
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Calculate standard IoU
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Create interpolated bounding box
    # The interpolated box is a linear interpolation between box1 and box2
    bi_x1 = (1 - interp_coe) * b1_x1 + interp_coe * b2_x1
    bi_y1 = (1 - interp_coe) * b1_y1 + interp_coe * b2_y1
    bi_x2 = (1 - interp_coe) * b1_x2 + interp_coe * b2_x2
    bi_y2 = (1 - interp_coe) * b1_y2 + interp_coe * b2_y2

    # Calculate IoU between interpolated box and box2
    inter_i = (torch.min(bi_x2, b2_x2) - torch.max(bi_x1, b2_x1)).clamp_(0) * (
        torch.min(bi_y2, b2_y2) - torch.max(bi_y1, b2_y1)
    ).clamp_(0)

    wi, hi = bi_x2 - bi_x1 + eps, bi_y2 - bi_y1 + eps
    w2, h2 = b2_x2 - b2_x1 + eps, b2_y2 - b2_y1 + eps

    union_i = wi * hi + w2 * h2 - inter_i + eps
    iou_i = inter_i / union_i

    # Final InterpIoU: IoU + IoU_interp - 1
    return iou + iou_i - 1


def d_interpiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    lv: float = 0.6,
    hv: float = 0.99,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate D-InterpIoU (Dynamic Interpolation IoU) between bounding boxes.

    D-InterpIoU is an adaptive version of InterpIoU that dynamically adjusts the interpolation
    coefficient based on the current IoU value. The interpolation coefficient is computed as:
    interp_coe = clamp(1 - IoU, min=lv, max=hv)

    This allows the method to adaptively adjust the interpolation based on how well the boxes
    are aligned, providing better gradient signals during training.

    Args:
        box1 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
                            Supports shapes like (4,), (N, 4), (B, N, 4), or (B, N, 1, 4).
        box2 (torch.Tensor): A tensor representing one or more bounding boxes, with the last dimension being 4.
                            Supports shapes like (4,), (M, 4), (B, M, 4), or (B, M, 1, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format (center coordinates).
                              If False, input boxes are in (x1, y1, x2, y2) format (corner coordinates).
                              Defaults to True.
        lv (float, optional): Lower bound for the dynamic interpolation coefficient. Defaults to 0.6.
        hv (float, optional): Upper bound for the dynamic interpolation coefficient. Defaults to 0.99.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): D-InterpIoU values with the same shape as the broadcast result of box1 and box2.

    Examples:
        >>> box1 = torch.tensor([[10, 10, 20, 20], [30, 30, 40, 40]])  # (N, 4) in xywh format
        >>> box2 = torch.tensor([[12, 12, 20, 20], [32, 32, 40, 40]])  # (M, 4) in xywh format
        >>> iou = d_interpiou(box1, box2, xywh=True, lv=0.6, hv=0.99)
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Calculate standard IoU
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Dynamic interpolation coefficient based on current IoU
    # When IoU is low, interp_coe is high (closer to box2)
    # When IoU is high, interp_coe is low (closer to box1)
    interp_coe = torch.clamp((1 - iou.detach()), min=lv, max=hv)

    # Create interpolated bounding box
    bi_x1 = (1 - interp_coe) * b1_x1 + interp_coe * b2_x1
    bi_y1 = (1 - interp_coe) * b1_y1 + interp_coe * b2_y1
    bi_x2 = (1 - interp_coe) * b1_x2 + interp_coe * b2_x2
    bi_y2 = (1 - interp_coe) * b1_y2 + interp_coe * b2_y2

    # Calculate IoU between interpolated box and box2
    inter_i = (torch.min(bi_x2, b2_x2) - torch.max(bi_x1, b2_x1)).clamp_(0) * (
        torch.min(bi_y2, b2_y2) - torch.max(bi_y1, b2_y1)
    ).clamp_(0)

    wi, hi = bi_x2 - bi_x1 + eps, bi_y2 - bi_y1 + eps
    w2, h2 = b2_x2 - b2_x1 + eps, b2_y2 - b2_y1 + eps

    union_i = wi * hi + w2 * h2 - inter_i + eps
    iou_i = inter_i / union_i

    # Final D-InterpIoU: IoU + IoU_interp - 1
    return iou + iou_i - 1

