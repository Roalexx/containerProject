import os
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def get_dicts(path):
    dataset_dicts = []
    fns = [s for s in os.listdir(os.path.join(path, 'images')) if s.endswith('.jpg')]
    for idx, fn in enumerate(fns):
        record = {}
        
        filename = os.path.join(path, 'images', fn)
        print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        data_fn = os.path.join(path, 'data', fn.split('.')[0]+'.txt')
        lines = open(data_fn, 'r').readlines()

        xs = [int(line.split()[0]) for line in lines]
        ys = [int(line.split()[1]) for line in lines]

        obj = {
            "bbox": [np.min(xs), np.min(ys), np.max(xs), np.max(ys)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [[(xs[i], ys[i]) for i in range(len(xs))]],
            "category_id": 0,
            "iscrowd": 0
            }
        record["annotations"] = [obj]
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "val"]:
    DatasetCatalog.register("container_" + d, lambda d=d: get_dicts(d))
    MetadataCatalog.get("container_" + d).set(thing_classes=["something"])


dataset_dicts = get_dicts("train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("container_train"), scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(vis.get_image()[:, :, ::-1])
    cv2.waitKey()


cfg = get_cfg()
cfg.OUTPUT_DIR = "./output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("container_train",)     # our training dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2     # number of parallel data loading workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")     # use pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2     # in 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # learning rate
cfg.SOLVER.MAX_ITER = 1000        # number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (mango)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
