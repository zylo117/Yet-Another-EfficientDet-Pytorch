# Author: Zylo117, Modified by K.M.Jeon

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import time
import os 
import torch
from torch.backends import cudnn

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from util.utils import preprocess_single_image, invert_affine, postprocess

compound_coef = 1
force_input_size = None  # set None to use default size
img_path = "./test/cross.mp4" # or 0 (camera id)

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

np.random.seed(5)
colors = np.random.uniform(0, 255, size=(len(obj_list), 3))

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

#Load EfficientDet Model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# Initialize Modules
with torch.no_grad():
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

# open video
vid = cv2.VideoCapture(img_path)

while vid.grab():
    ret, frame = vid.retrieve()
    if not ret:
        break  
    # BGR 2 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # preprocss for single image
    framed_img, framed_meta = preprocess_single_image(frame, max_size=input_size)

    #  detection
    inference_start_time = time.time()

    # convert to tensor
    if use_cuda:
        x = torch.stack([torch.from_numpy(framed_img).cuda()], 0)
    else:
        x = torch.stack([torch.from_numpy(framed_img)], 0)
    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
    inference_end_time = time.time()
    #logging.info("\t Inference time: %s ", datetime.timedelta(seconds=inference_end_time - inference_start_time))

    # detection over

    # log FPS on frame
    ...


    # get detection bbox
    out = invert_affine([framed_meta], out)
    pred = out[0]

    # reverse to BGR
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # draw bbox on frame and display
    for j in range(len(pred['rois'])):
        (x1, y1, x2, y2) = pred['rois'][j].astype(np.int)
        cv2.rectangle(img, (x1, y1), (x2, y2), colors[pred['class_ids'][j]], 2)
        obj = obj_list[pred['class_ids'][j]]
        score = float(pred['scores'][j])
        cv2.putText(img, '{}, {:.3f}'.format(obj, score),
                    (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[pred['class_ids'][j]], 1)

    cv2.imshow('Video', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows() 
        os._exit(0)