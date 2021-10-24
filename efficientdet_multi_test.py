# Author: firsteast

"""
Multiple Image Inference Script of EfficientDet-Pytorch
"""
import time, os
import torch
from torch.backends import cudnn
from matplotlib import colors
from tqdm import tqdm

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

# set informations of parameters for run code
compound_coef = 0
force_input_size = None  # set None to use default size
IMAGE_PATH = 'path/to/test_image' # path where test images are saved
imgs = os.listdir(IMAGE_PATH)
MODEL_PATH = 'path/to/model' # path where model(.pth) is saved
RESULT_PATH = 'path/to/save/results' # path to save test images

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['fingertips', 'object']

color_list = standard_to_bgr(STANDARD_COLORS)
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536] 
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# create directory for saving result path
if not os.path.isdir(RESULT_PATH):
    os.mkdir(RESULT_PATH)
    print(RESULT_PATH + ' created')
if not os.path.isdir(RESULT_PATH + '/fail'):
    os.mkdir(RESULT_PATH + '/fail')
    print(RESULT_PATH + '/fail created')

model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                            ratios=anchor_ratios, scales=anchor_scales)
model.load_state_dict(torch.load(f'{MODEL_PATH}/efficientdet-d{compound_coef}_499_6500.pth', map_location='cpu'))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

def display(preds, imgs, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            cv2.imwrite(f'{RESULT_PATH}/fail/{img_name}_d{compound_coef}.jpg', imgs[i])
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(f'{RESULT_PATH}/{img_name}_d{compound_coef}.jpg', imgs[i])

# test
for img in tqdm(imgs):
    img_path = IMAGE_PATH + img
    img_name = img.split('_')[0]

    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        regressBoxes = BBoxTransform()
        clipBoxes = ClipBoxes()

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    out = invert_affine(framed_metas, out)
    display(out, ori_imgs, imshow=False, imwrite=True)

    print('running speed test...')
    with torch.no_grad():
        print(img_name + ' test1: model inferring and postprocessing')
        print('inferring image for 10 times...')
        t1 = time.time()
        for _ in range(10):
            _, regression, classification, anchors = model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, iou_threshold)
            out = invert_affine(framed_metas, out)

        t2 = time.time()
        tact_time = (t2 - t1) / 10
        print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

        # uncomment this if you want a extreme fps test
        # print('test2: model inferring only')
        # print('inferring images for batch_size 32 for 10 times...')
        # t1 = time.time()
        # x = torch.cat([x] * 32, 0)
        # for _ in range(10):
        #     _, regression, classification, anchors = model(x)
        #
        # t2 = time.time()
        # tact_time = (t2 - t1) / 10
        # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
