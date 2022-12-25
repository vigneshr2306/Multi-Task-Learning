import open3d as o3d
from base64 import b64encode
# from IPython.display import HTML
import matplotlib.colors as co
import matplotlib.cm as cm
import glob
from lib.network.hydranet_autonomous_car import HydraNet
from torch.autograd import Variable
import torch.nn.functional as F
import math
import torch.nn as nn
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# %matplotlib inline
# !export PYTHONPATH = .

hydranet = HydraNet()
hydranet.define_mobilenet()
hydranet.define_lightweight_refinenet()
if torch.cuda.is_available():
    _ = hydranet.cuda()
_ = hydranet.eval()

# ckpt = torch.load('models/ExpNYUDKITTI_joint.ckpt')
ckpt = torch.load('models/ExpKITTI_joint.ckpt')
hydranet.load_state_dict(ckpt['state_dict'])

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))


def prepare_img(img):
    return (img * IMG_SCALE - IMG_MEAN) / IMG_STD


# Pre-processing and post-processing constants #
CMAP = np.load('cmaps/cmap_kitti.npy')
NUM_CLASSES = 6

print(CMAP)

images_files = glob.glob(
    'inference_data_kitti/KITTI/image_2/0001/*.png')
idx = np.random.randint(0, len(images_files))

img_path = images_files[idx]
img = np.array(Image.open(img_path))
plt.imshow(img)
plt.show()


def pipeline(img):
    with torch.no_grad():
      # Converting to the correct colorspace
        img_var = Variable(torch.from_numpy(prepare_img(img).transpose(
            2, 0, 1)[None]), requires_grad=False).float()
        if torch.cuda.is_available():
            img_var = img_var.cuda()
        segm, depth = hydranet(img_var)
        segm = cv2.resize(segm[0, :NUM_CLASSES].cpu().data.numpy().transpose(1, 2, 0),
                          img.shape[:2][::-1],
                          interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth[0, 0].cpu().data.numpy(),
                           img.shape[:2][::-1],
                           interpolation=cv2.INTER_CUBIC)
        segm = CMAP[segm.argmax(axis=2)].astype(np.uint8)
        depth = np.abs(depth)
        return depth, segm


depth, segm = pipeline(img)

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 20))
ax1.imshow(img)
ax1.set_title('Original', fontsize=30)
ax2.imshow(segm)
ax2.set_title('Predicted Segmentation', fontsize=30)
ax3.imshow(depth, cmap="plasma", vmin=0, vmax=80)
ax3.set_title("Predicted Depth", fontsize=30)
plt.show()

print(img.shape)
print(depth.shape)
print(segm.shape)


def depth_to_rgb(depth):
    normalizer = co.Normalize(vmin=0, vmax=80)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='plasma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    return colormapped_im


depth_rgb = depth_to_rgb(depth)
print(depth_rgb.shape)
plt.imshow(depth_rgb)
plt.show()

print(img.shape)
print(depth_rgb.shape)
print(segm.shape)
new_img = np.vstack((img, segm, depth_rgb))
plt.imshow(new_img)
plt.show()

video_files = sorted(glob.glob(
    "inference_data_kitti/KITTI/image_2/0001/*.png"))

# Build a HydraNet
hydranet = HydraNet()
hydranet.define_mobilenet()
hydranet.define_lightweight_refinenet()

# Set the Model to Eval on GPU
if torch.cuda.is_available():
    _ = hydranet.cuda()
_ = hydranet.eval()

# Load the Weights
ckpt = torch.load('models/ExpKITTI_joint.ckpt')
hydranet.load_state_dict(ckpt['state_dict'])

# Run the pipeline

result_video = []
for idx, img_path in enumerate(video_files):
    image = np.array(Image.open(img_path))
    h, w, _ = image.shape
    depth, seg = pipeline(image)
    result_video.append(cv2.cvtColor(cv2.vconcat(
        [image, seg, depth_to_rgb(depth)]), cv2.COLOR_BGR2RGB))

image = np.array(Image.open(img_path))
h, w, _ = image.shape
out = cv2.VideoWriter(
    'output/out1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (w, 3*h))

for i in range(len(result_video)):
    out.write(result_video[i])
out.release()
