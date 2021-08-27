# Author: Zylo117

"""
COCO-Style Evaluations

put images here datasets/your_project_name/val_set_name/*.jpg
put annotations here datasets/your_project_name/annotations/instances_{val_set_name}.json
put weights here /path/to/your/weights/*.pth
change compound_coef

ADDITIONAL INSTRUCTIONS AFTER INTRODUCING BATCH WISE AND MULTI GPU EVALUATION:
change the batch_size
change the number of GPUs in yml file

"""

import json
import os

import argparse
import torch
import yaml
import math
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess_eval, invert_affine, postprocess, boolean_string

from utils.utils import replace_w_sync_bn, CustomDataParallel
from utils.sync_batchnorm import patch_replication_callback

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
args = ap.parse_args()

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = Params(f'projects/{args.project}.yml')
obj_list = params.obj_list

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    results = []

    no_of_batches = math.ceil(len(image_ids)/args.batch_size)
    batches = no_of_batches*[0]

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for batch in tqdm(range(no_of_batches)):
        batches[batch] = []
        start = batch*args.batch_size
        if batch == no_of_batches-1:
            end = len(image_ids)
        else:
            end = (batch*args.batch_size) + args.batch_size
        for image_id in range(start, end):
            image_info = coco.loadImgs(image_id)[0]
            image_path = img_path + image_info['file_name']
            batches[batch].append(image_path)

        ori_imgs, framed_imgs, framed_metas = preprocess_eval(batches[batch], max_size=input_sizes[compound_coef], mean=params.mean, std=params.std)
        x = torch.tensor(framed_imgs[0:args.batch_size])

        if params.num_gpus == 1:
            x = x.cuda(gpu)
            if params.num_gpus > 1:
                x = CustomDataParallel(x, params.num_gpus)
                if use_sync_bn:
                    patch_replication_callback(x)

            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)
        
        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
                            
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0:args.batch_size]
        
        for image in range(args.batch_size):
            image_id = (batch*args.batch_size) + image

            if batch == no_of_batches-1 and image_id == len(image_ids):
                break

            scores = preds[image]['scores']
            class_ids = preds[image]['class_ids']
            rois = preds[image]['rois']

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
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    # write output
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox \n\nOverall:\n')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params.val_set
    VAL_GT = f'datasets/{params.project_name}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'datasets/{params.project_name}/{SET_NAME}/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        if params.num_gpus > 1 and args.batch_size // params.num_gpus < 4:
            model.apply(replace_w_sync_bn)
            use_sync_bn = True
        else:
            use_sync_bn = False

        if params.num_gpus > 0:
            model = model.cuda(gpu)
            if params.num_gpus > 1:
                model = CustomDataParallel(model, params.num_gpus)
                if use_sync_bn:
                    patch_replication_callback(model)

        model.requires_grad_(False)
        model.eval()

        evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
