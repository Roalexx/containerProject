import os
import cv2
import numpy as np 
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from train import get_dicts

cfg = get_cfg()
cfg.OUTPUT_DIR = "./output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("container_train",)     # our training dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2     # number of parallel data loading workers
cfg.SOLVER.IMS_PER_BATCH = 2     # in 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # learning rate
cfg.SOLVER.MAX_ITER = 1000        # number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (mango)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("container_val", )
predictor = DefaultPredictor(cfg)



dataset_dicts = get_dicts("val")
for d in dataset_dicts:    
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    mask = outputs["instances"].to("cpu").pred_masks[0].numpy()
    mask = np.dstack([mask]*3)
    img = img * mask
    cv2.imshow(img)
    cv2.waitKey()