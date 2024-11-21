    

# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union
import os 

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results




def results2json(self, results: Sequence[dict],
                     outfile_prefix: str) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        bbox_json_results = []
        segm_json_results = [] if 'masks' in results[0] else None
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            # bbox results
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(scores[i])
                data['category_id'] = self.cat_ids[label]
                bbox_json_results.append(data)

            if segm_json_results is None:
                continue

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(mask_scores[i])
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                segm_json_results.append(data)

        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])

        if segm_json_results is not None:
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            dump(segm_json_results, result_files['segm'])

        return 


import tqdm
import argparse
import os
import json
import shutil
import numpy as np
import cv2
import pdb
import datetime
from scipy.io import loadmat
import pycocotools.mask as mask_util
from pycocotools import mask
import cv2

# def coco_ann( l, h, w, x, y, cat_id,image_id, segm):


#             coco_annotation = {}
#             coco_annotation['image_id'] = image_id
#             coco_annotation['id'] = l + 1
#             coco_annotation['width'] = w
#             coco_annotation['height'] = h
#             coco_annotation['category_id'] = cat_id
#             # bbox_pred = np.asarray(instances.pred_boxes[i].tensor[0])
#             # bbox_pred = [bbox_pred[0], bbox_pred[1], bbox_pred[2] - bbox_pred[0], bbox_pred[3] - bbox_pred[1]]
#             # compress to RLE format
#             segm_pred = mask.encode(np.asfortranarray( segm).astype('uint8'))
#             ############## POLYGON format saving  ########################
#             # maskedArr = mask.decode(segm_pred)
#             # contours,_ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#             # segment = []
#             # valid_poly = 0
#             # for contour in contours:
#             #     if contour.size >= 6:
#             #         segment.append(contour.astype(float).flatten().tolist())
#             #         valid_poly += 1
#             #         if valid_poly == 0:
#             #             raise ValueError
#             # coco_annotation['segmentation'] = segment

#             ################ POLYGON format saving#########################
#             # maskedArr = mask.decode(segm_pred)
#             ########### RLE format saving #########
#             coco_annotation['segmentation'] = segm_pred
#             area = mask_util.area(segm_pred).item()
#             coco_annotation['area'] = int(area)
#             if isinstance(segm_pred, dict):  # RLE
#                 counts = segm_pred["counts"]
#                 if not isinstance(counts, str):
#                     # make it json-serializable
#                     coco_annotation['segmentation']['counts'] = counts.decode("ascii")
#             ############ RLE format saving #########

#             x1 = np.min(x)
#             y1 = np.min(y)
#             x2 = np.max(x)
#             y2 = np.max(y)
#             coco_bbox = [ x1, y1, x2-x1, y2-y1]
#             coco_annotation['bbox'] = list(int(np.round(x)) for x in coco_bbox)
#             coco_annotation["iscrowd"] = 0 # Polygon format uses 0 but rle uses 1, slightly confused here


#             return coco_annotation

def xyxy2xywh( bbox: np.ndarray) -> list:
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    _bbox: List = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0],
        _bbox[3] - _bbox[1],
    ]

def main(image_path='', annotation_path=''):

        coco_annotations = []
        coco_images = []
        c = 0
        categories = [{"id": 1, "name": "Cerebellum", "supercategory": "shape"}, {"id": 2, "name": "Arachnoid", "supercategory": "shape"},{"id": 3, "name": "CN8", "supercategory": "shape"}, {"id": 4, "name": "CN5", "supercategory": "shape"}, {"id": 5, "name": "CN7", "supercategory": "shape"}, {"id": 6, "name": "CN_9_10_11", "supercategory": "shape"},{"id": 7, "name": "SCA", "supercategory": "shape"}, {"id": 8, "name": "AICA", "supercategory": "shape"}, {"id": 9, "name": "SuperiorPetrosalVein", "supercategory": "shape"}, {"id": 10, "name": "Labrynthine", "supercategory": "shape"}, {"id": 11, "name": "Vein", "supercategory": "shape"}, {"id": 12, "name": "Brainstem", "supercategory": "shape"},{"id": 1001, "name": "Suction", "supercategory": "shape"}, {"id": 1002, "name": "Bovie", "supercategory": "shape"},{"id": 1003, "name": "Bipolar", "supercategory": "shape"}, {"id": 1004, "name": "Forcep", "supercategory": "shape"}, {"id": 1005, "name": "BluntProbe", "supercategory": "shape"}, {"id": 1006, "name": "Drill", "supercategory": "shape"}, {"id": 1007, "name": "Kerrison", "supercategory": "shape"}, {"id": 1008, "name": "Cottonoid", "supercategory": "shape"}, {"id": 1009, "name": "Scissors", "supercategory": "shape"}, {"id": 1012, "name": "Unknown", "supercategory": "shape"}]

        for filename in os.listdir(annotation_path):

            framename = filename.split('.')[0]
            frame_num = framename.split('.')[0][-4:]

            json_file_path = os.path.join(annotation_path,framename + '.json')
            with open(json_file_path, 'r') as f:
                result = json.load(f)

            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']

            # segm results
            masks = result['masks']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = int(frame_num)

                x1 = bboxes[i][0]
                y1 = bboxes[i][1]
                x2 = bboxes[i][2]
                y2 = bboxes[i][3]
                coco_bbox = [ x1, y1, x2-x1, y2-y1]
                data['bbox'] = list(int(np.round(x)) for x in coco_bbox)
                data['score'] = float(scores[i])
                if data['score']<=0.7:
                     continue
                data['category_id'] = categories[label]["id"]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                data['segmentation'] = masks[i]
                data['id'] = c + 1
                c = c + 1
                data["iscrowd"] =0 
                coco_annotations.append(data)



                #INSERT CODE TO GENERATE COCO FORMAT FOR PREDICTIONS
            coco_image = {
                "id": int(frame_num),
                "width":1920 ,
                "height": 1080,
                "file_name": framename + '.jpg',
            }

            coco_images.append(coco_image)




        return coco_images, coco_annotations




if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--images", default="")
    # parser.add_argument("--annotations",default="")
    # parser.add_argument("--output_data")
    # args = parser.parse_args()

    # os.makedirs(args.output_data, exist_ok=True)

    images = "/home/nehal/Data/Neuro/Videos/VID002B_0.0.0_0.4.11"
    annotations = "/home/nehal/code/mmdetection/outputs/preds"

    coco_images, coco_annotations = main(images, annotations)
# save coco json

    dataset_name = 'Surgery_COCO'
    output_file = 'vid511B_mask2former_swintransformerfixed_coco_score0.7.json'


    # These categories are instance specific
    categories = [{"id": 1, "name": "Cerebellum", "supercategory": "shape"}, {"id": 2, "name": "Arachnoid", "supercategory": "shape"},{"id": 3, "name": "CN8", "supercategory": "shape"}, {"id": 4, "name": "CN5", "supercategory": "shape"}, {"id": 5, "name": "CN7", "supercategory": "shape"}, {"id": 6, "name": "CN_9_10_11", "supercategory": "shape"},{"id": 7, "name": "SCA", "supercategory": "shape"}, {"id": 8, "name": "AICA", "supercategory": "shape"}, {"id": 9, "name": "SuperiorPetrosalVein", "supercategory": "shape"}, {"id": 10, "name": "Labrynthine", "supercategory": "shape"}, {"id": 11, "name": "Vein", "supercategory": "shape"}, {"id": 12, "name": "Brainstem", "supercategory": "shape"},{"id": 1001, "name": "Suction", "supercategory": "shape"}, {"id": 1002, "name": "Bovie", "supercategory": "shape"},{"id": 1003, "name": "Bipolar", "supercategory": "shape"}, {"id": 1004, "name": "Forcep", "supercategory": "shape"}, {"id": 1005, "name": "BluntProbe", "supercategory": "shape"}, {"id": 1006, "name": "Drill", "supercategory": "shape"}, {"id": 1007, "name": "Kerrison", "supercategory": "shape"}, {"id": 1008, "name": "Cottonoid", "supercategory": "shape"}, {"id": 1009, "name": "Scissors", "supercategory": "shape"}, {"id": 1012, "name": "Unknown", "supercategory": "shape"}]
    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {"info": info, "images": coco_images, "categories": categories, "licenses": None}
    if len(coco_annotations) > 0:

        coco_dict["annotations"] = coco_annotations

        # logger.info(f"Caching COCO format annotations at '{output_file}' ...")
    with open(output_file, 'w') as train_file:
        json.dump(coco_dict, train_file)