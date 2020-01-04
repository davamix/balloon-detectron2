import detectron2
import numpy as np

#import cv2
import matplotlib.pyplot as plt
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from torchvision import transforms


# Load image
#im = cv2.imread("./input.jpg")
im = Image.open("./input.jpg")
# Enable to show window images on console
#cv2.startWindowThread()
#cv2.namedWindow("preview")

# Convert Image to numpy array expected by the predictor
im_array = np.array(im)
print(im_array.shape)


cfg = get_cfg()

# Load model
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"

predictor = DefaultPredictor(cfg)
outputs = predictor(im_array)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# Visualize boxes
v = Visualizer(im_array[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save predictions
fig = plt.figure()
plt.imshow(v.get_image()[:,:,::-1])
fig.savefig("./prediction.jpg")
#cv2.imshow("preview", v.get_image()[:, :, ::-1])
#cv2.waitKey()
