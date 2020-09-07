import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import numpy as np
import os, sys, random, math, coco, utils, visualize, torch, glob
import skimage.io
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
print(MODEL_DIR)
# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()
# Load weights trained on MS-COCO
model.load_state_dict(torch.load(COCO_MODEL_PATH))

pathlist = glob.glob("imgs/*.jpg")
for i, fname in enumerate(pathlist):
    # Read
    image = Image.open(fname)
    im = np.asarray(image)
    # Run detection
    results = model.detect([im])
    # Visualize results
    r = results[0]
    x1, y1 ,x2, y2 = r['rois'][0]
    # Create rectangle image 
    img1 = ImageDraw.Draw(image)   
    img1.rectangle([(y1, x1), (y2, x2)], outline ="red")
    if not os.path.exists("results"):
        os.mkdir("results")
    fname = os.path.splitext(fname)[0].split("/")[-1]
    image.save(os.path.join("results",fname+".jpg"))
    #plt.imshow(image)
    #plt.show()
