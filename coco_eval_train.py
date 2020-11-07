# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/annotations/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

"""

import json
import os

import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]

compound_coef = 0
nms_threshold = 0.2
use_cuda = True
use_float16 = False
override_prev_results = True
project_name = "proj"
confidence = 0.05


params = ""
obj_list = ""




def evaluate_coco(img_path, set_name, image_ids, coco, model,filepath,threshold=confidence):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']
        #print(image_path)

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda()
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model.model(x)
        #print(regression,classification)
        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        #print(preds)
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        print('the model does not provide any valid output, check model architecture and the data input')

    # write output
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    if len(results) <= 0:
        return False
    return True


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def eval_valid(data_path,compound_coef,model,nms_threshold,use_cuda,use_float16,override_prev_results,project_name,confidence=0.05):
    
    listOfGlobals = globals()
    listOfGlobals['compound_coef'] = compound_coef
    listOfGlobals['nms_threshold'] = nms_threshold
    listOfGlobals['use_cuda'] = use_cuda
    listOfGlobals['use_float16'] = use_float16
    listOfGlobals['override_prev_results'] = override_prev_results
    listOfGlobals['project_name'] = project_name
    listOfGlobals['params'] = yaml.safe_load(open(f'{project_name}.yml'))
    listOfGlobals['obj_list'] = listOfGlobals['params']['obj_list']
    listOfGlobals['confidence'] = confidence
    params = listOfGlobals['params']
    SET_NAME = params['val_set']
    VAL_GT = f'{data_path}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{data_path}/{params["project_name"]}/{SET_NAME}/'
    VAL_PRED = f'{data_path}/{params["project_name"]}/annotations/{SET_NAME}_bbox_results.json'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    is_pred_ok = True
    if override_prev_results or not os.path.exists(VAL_PRED):
        is_pred_ok = evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model,VAL_PRED)

    if is_pred_ok:
        _eval(coco_gt, image_ids, VAL_PRED)
    else:
        print("NO VALID DETECTIONS FOUND.")
