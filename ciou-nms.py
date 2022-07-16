import math

import numpy as np
import torch
import torchvision


def non_max_suppression(prediction, conf_thres=0.1, nms_thres=0.6, multi_cls=True):
    min_wh, max_wh = 1, 4096  # (pixels) 宽度和高度的大小范围 [min_wh, max_wh]
    output = [None] * len(prediction)  # batch_size个output  存放最终筛选后的预测框结果
    for image_i, pred in enumerate(prediction):
        pred = pred[pred[:, 4] > conf_thres]
        pred = pred[(pred[:, 2:4] > min_wh).all(1) & (pred[:, 2:4] < max_wh).all(1)]
        if len(pred) == 0:
            continue
        pred[..., 5:] *= pred[..., 4:5]
        box = xywh2xyxy(pred[:, :4])
        if multi_cls or conf_thres < 0.01:
            i, j = (pred[:, 5:] > conf_thres).nonzero(as_tuple=False).t()
            pred = torch.cat((box[i], pred[i, j + 5].unsqueeze(1), j.float().unsqueeze(1)), 1)
        else:
            conf, j = pred[:, 5:].max(1)
            pred = torch.cat((box, conf.unsqueeze(1), j.float().unsqueeze(1)), 1)[conf > conf_thres]
        if len(pred) == 0:
            continue
        pred = pred[pred[:, 4].argsort(descending=True)]
        det_max = []
        cls = pred[:, -1]
        for c in cls.unique():
            dc = pred[cls == c]
            n = len(dc)
            if n == 1:
                det_max.append(dc)
                continue
            elif n > 500:
                dc = dc[:500]
            while dc.shape[0]:
                det_max.append(dc[:1])
                if len(dc) == 1:
                    break
                diou = bbox_iou(dc[0], dc[1:], CIoU=True)
                dc = dc[1:][diou < nms_thres]
        if len(det_max):
            det_max = torch.cat(det_max)
            output[image_i] = det_max[(-det_max[:, 4]).argsort()]
    return output


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU or SIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                w_dis = torch.pow(b1_x2 - b1_x1 - b2_x2 + b2_x1, 2)
                h_dis = torch.pow(b1_y2 - b1_y1 - b2_y2 + b2_y1, 2)
                cw2 = torch.pow(cw, 2) + eps
                ch2 = torch.pow(ch, 2) + eps
                return iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2)
            else:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                iou = iou - 0.5 * (distance_cost + shape_cost)
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return y
