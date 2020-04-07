# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import torch
from torchvision.ops import nms

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes

compound_coef = 0
force_input_size = 1920  # set None to use default size

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # background is ignored

# use it only in COCO dataset
label_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
             9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
             18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
             27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
             37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
             46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
             54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
             62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
             74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
             82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w

    if c > 1:
        canvas[:new_h, :new_w] = image
    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h


ori_img = cv2.imread('test/img.png')
x = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
x = x / 255
x = x - (0.485, 0.456, 0.406)
x = x / (0.229, 0.224, 0.225)

# tf bilinear interpolation is different from any other's, just make do
input_size = 512 + compound_coef * 128 if force_input_size is None else force_input_size
imgs_meta = aspectaware_resize_padding(x, input_size, input_size, cv2.INTER_LINEAR)
x = imgs_meta[0]
framed_metas = imgs_meta[1:]
x = torch.from_numpy(x).cuda().to(torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

# x = torch.from_numpy(np.load('test/tf.npy')).cuda().permute((0,3,1,2))
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=90).cuda()
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

with torch.no_grad():
    features, regression, classification, anchors = model(x)

threshold = 0.2
iou_threshold = 0.2

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

transformed_anchors = regressBoxes(anchors, regression)
transformed_anchors = clipBoxes(transformed_anchors, x)
scores = torch.max(classification, dim=2, keepdim=True)[0]
scores_over_thresh = (scores > threshold)[:, :, 0]

out = []
for i in range(x.shape[0]):
    classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
    transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
    scores_per = scores[i, scores_over_thresh[i, :], ...]
    anchors_nms_idx = nms(transformed_anchors_per, scores_per[:, 0], iou_threshold=iou_threshold)

    scores_, classes_ = classification_per[:, anchors_nms_idx].max(dim=0)
    boxes_ = transformed_anchors_per[anchors_nms_idx, :]

    out.append({
        'rois': boxes_.cpu().numpy(),
        'class_ids': classes_.cpu().numpy(),
        'scores': scores_.cpu().numpy(),
    })


def invert_affine(metas, preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
            preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
            preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[label_map[preds[i]['class_ids'][j] + 1] - 1]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        cv2.imshow('img', imgs[i])
        cv2.waitKey(0)
        cv2.imwrite(f'test/img_inferred_d{compound_coef}_this_repo.jpg', imgs[i])


out = invert_affine([framed_metas], out)
display(out, [ori_img])
