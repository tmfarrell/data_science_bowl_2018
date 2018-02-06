"""
Mask R-CNN
Configurations and data loading code for Data Science Bowl 2018 dataset.

https://www.kaggle.com/c/data-science-bowl-2018

-----------------------------------------------------------
adapted from Mask-RCNN/coco.py

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
"""

import os
import sys
import time
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imshow
'''
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import zipfile
import urllib.request
import shutil
'''
sys.path.append('/Users/tfarrell/Dropbox/projects/mls_data_science_bowl2018/Mask-RCNN/')
from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()
DATA_DIR = '/Users/tfarrell/Data/data_science_bowl_2018'

# Path to MS COCO trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  Configuration
############################################################

class NucleiConfig(Config): 
    """
    Configuration for training on Broad Nuclei
    (Data Science Bowl 2018) dataset. 
    """
    NAME = "nuclei"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 1     # nuclei or background
    IMG_WIDTH = 720
    IMG_HEIGHT = 720
    IMG_CHANNELS = 3
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 25
    
############################################################
#  Dataset
############################################################

class NucleiDataset(utils.Dataset): 
    def load_dataset(self, data_dir, subset, img_height, img_width): 
        # add nucleus class (the only class besides background)
        self.add_class("nuclei", 1, "nucleus")
        # load img ids
        if subset == 'train': 
            img_ids = [l.strip() for l in open(os.path.join(data_dir, 'train_ids.txt')).readlines()]
            data_dir = os.path.join(data_dir, 'stage1_train')
        elif subset == 'validate': 
            img_ids = [l.strip() for l in open(os.path.join(data_dir, 'validate_ids.txt')).readlines()]
            data_dir = os.path.join(data_dir, 'stage1_train')
        elif subset == 'test': 
            data_dir = os.path.join(data_dir, 'stage1_test')
            img_ids = [id_ for id_ in os.listdir(data_dir) if not '.' in id_]
        # add imgs
        for img_id in img_ids: 
            # get img path
            img_path = os.path.join(data_dir, img_id, 
                                    ('images' if subset in ['train','validate'] else ''),
                                    img_id + '.png')
            # get mask paths 
            if subset in ['train','validate']:
                mask_paths = [os.path.join(data_dir, img_id, 'masks', mask_file) 
                              for mask_file in os.listdir(os.path.join(data_dir, img_id, 'masks'))]
                mask_ids = [mask_path[mask_path.rfind('/')+1:mask_path.rfind('.')] for mask_path in mask_paths]
            else: 
                mask_ids = None
                mask_paths = None
            self.add_image("nuclei", image_id=img_id, path=img_path, 
                           width=img_width, height=img_height, 
                           mask_paths=mask_paths, mask_ids=mask_ids)
        return

    def load_image(self, image_id, img_height, img_width, img_channels): 
        info = self.image_info[image_id]
        img = imread(info['path'])[:,:,:img_channels]
        img = resize(img, (img_height,img_width), mode='constant', preserve_range=True)
        return(img)

    def load_mask(self, image_id): 
        info = self.image_info[image_id]
        mask_paths = info['mask_paths']
        mask = np.zeros([info['height'], info['width'], len(mask_paths)], dtype=np.uint8)
        for i, mask_path in enumerate(mask_paths): 
            mask_ = imread(mask_path)
            mask_ = np.expand_dims(resize(mask_, (info['height'], info['width']), 
                                          mode='constant', preserve_range=True), axis=-1)
            mask[:, :, i:i+1] = mask_
        class_ids = np.array([1] * len(mask_paths))     # all masks are nuclei
        return mask, class_ids.astype(np.int32)


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False,
                        default=DEFAULT_DATASET_YEAR,
                        metavar="<year>",
                        help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False,
                        default=False,
                        metavar="<True|False>",
                        help='Automatically download and unzip MS-COCO files (default=False)',
                        type=bool)
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Year: ", args.year)
    print("Logs: ", args.logs)
    print("Auto Download: ", args.download)

    # Configurations
    if args.command == "train":
        config = CocoConfig()
    else:
        class InferenceConfig(CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_coco(args.dataset, "train", year=args.year, auto_download=args.download)
        dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_coco(args.dataset, "minival", year=args.year, auto_download=args.download)
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print("Running COCO evaluation on {} images.".format(args.limit))
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
