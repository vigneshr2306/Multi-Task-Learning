#!/usr/bin/env python
# coding: utf-8

# # HydraNet Home Robot Project 🤖
#
# In this workshop, you're going to learn how to train a Neural Network that does **real-time semantic segmentation and monocular depth prediction**.
#
# ![](https://d3i71xaburhd42.cloudfront.net/435d4b5c30f10753d277848a17baddebd98d3c31/2-Figure1-1.png)
#
# The Model is [a Multi-Task Learning algorithm designed by Vladimir Nekrasov](https://arxiv.org/pdf/1809.04766.pdf). The entire work is based on the **DenseTorch Library**, that you can find and use [here](https://github.com/DrSleep/DenseTorch). <p>
#
# * The **KITTI Dataset only has 200 examples of segmentation**. Therefore, the authors used a technique called Knowledge Distillation and finetuned using the Cityscape dataset.<p>
#
# * 👉 In our case, we'll use another dataset called the [NYUDv2 Dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). **It contains 1449 annotated images for depth and segmentation**, which makes our life much simpler.


import operator
from lib.utils.utils import MeanIoU, RMSE
from tqdm import tqdm
from lib.utils.utils import AverageMeter
import logging
import json
from lib.utils.model_helpers import Saver, load_state_dict
from lib.utils.utils import InvHuberLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from lib.utils.utils import Normalise, RandomCrop, ToTensor, RandomMirror
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.network.encoder import MobileNetv2
from lib.network.decoder import MTLWRefineNet
from lib.datasets import HydraNetDataset

# # 1 — Dataset
# Let's begin with importing our data, and visualizing it.

# ## Load and Visualize the Dataset


import glob

depth = sorted(
    glob.glob("/home/vicky/Coding/Projects/nyud/depth/*.png"))
seg = sorted(glob.glob("/home/vicky/Coding/Projects/nyud/masks/*.png"))
images = sorted(glob.glob("/home/vicky/Coding/Projects/nyud/rgb/*.png"))


print(len(images))
print(len(depth))
print(len(seg))


# Since our dataset is a bit "special", we'll need a Color Map to read it.


CMAP = np.load('cmaps/cmap_nyud.npy')
print(len(CMAP))


idx = np.random.randint(0, len(seg))

f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 40))
ax0.imshow(np.array(Image.open(images[idx])))
ax0.set_title("Original")
ax1.imshow(np.array(Image.open(depth[idx])), cmap="plasma")
ax1.set_title("Depth")
ax2.imshow(CMAP[np.array(Image.open(seg[idx]))])
ax2.set_title("Segmentation")
plt.show()


print(np.unique(np.array(Image.open(seg[idx]))))
print(len(np.unique(np.array(Image.open(seg[idx])))))


# ## Getting the DataLoader
#
# When training a model, 2 elements are going to be very important (compared to the last workshop):
#
# *   The Dataset
# *   The Training Loop, Loss, etc
#
# We already know how to design the model that does join depth and segmentation, so we only need to know how to train it!


data_file = "train_list_depth.txt"

with open(data_file, "rb") as f:
    datalist = f.readlines()
datalist = [x.decode("utf-8").strip("\n").split("\t") for x in datalist]

root_dir = "/home/vicky/Coding/Projects/nyud"
masks_names = ("segm", "depth")

print(datalist[0])


abs_paths = [os.path.join(root_dir, rpath) for rpath in datalist[0]]
abs_paths


img_arr = np.array(Image.open(abs_paths[0]))

plt.imshow(img_arr)
plt.show()


masks_names = ("segm", "depth")

for mask_name, mask_path in zip(masks_names, abs_paths[1:]):
    mask = np.array(Image.open(mask_path))
    print(mask_name)
    plt.imshow(mask)
    plt.show()


# ### Normalization — Will be common to all images
#


img_scale = 1.0 / 255
depth_scale = 5000.0

img_mean = np.array([0.485, 0.456, 0.406])
img_std = np.array([0.229, 0.224, 0.225])

normalise_params = [img_scale, img_mean.reshape(
    (1, 1, 3)), img_std.reshape((1, 1, 3)), depth_scale, ]

transform_common = [Normalise(*normalise_params), ToTensor()]


# ### Transforms


crop_size = 400
transform_train = transforms.Compose(
    [RandomMirror(), RandomCrop(crop_size)] + transform_common)
transform_val = transforms.Compose(transform_common)


# ### DataLoader


train_batch_size = 4
val_batch_size = 4
train_file = "train_list_depth.txt"
val_file = "val_list_depth.txt"


# TRAIN DATALOADER
trainloader = DataLoader(
    HydraNetDataset(train_file, transform=transform_train,),
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# VALIDATION DATALOADER
valloader = DataLoader(
    HydraNetDataset(val_file, transform=transform_val,),
    batch_size=val_batch_size,
    shuffle=False, num_workers=4,
    pin_memory=True,
    drop_last=False,
)


# # 2 — Creating the HydraNet
# We now have 2 DataLoaders: one for training, and one for validation/test. <p>
#
# In the next step, we're going to define our model, following the paper [Real-Time Joint Semantic Segmentation and Depth Estimation Using Asymmetric Annotations](https://arxiv.org/pdf/1809.04766.pdf) —— If you haven't read it yet, now is the time.
# <p>
#
# > ![](https://d3i71xaburhd42.cloudfront.net/435d4b5c30f10753d277848a17baddebd98d3c31/2-Figure1-1.png)
#
# Our model takes an input RGB image, make it go through an encoder, a lightweight refinenet decoder, and then has 2 heads, one for each task.<p>
# Things to note:
# * The only **convolutions** we'll need will be 3x3 and 1x1
# * We also need a **MaxPooling 5x5**
# * **CRP-Blocks** are implemented as Skip-Connection Operations
# * **Each Head is made of a 1x1 convolution followed by a 3x3 convolution**, only the data and the loss change there
#

# ## Building the Encoder — A MobileNetv2
# ![](https://iq.opengenus.org/content/images/2020/11/conv_mobilenet_v2.jpg)


encoder = MobileNetv2()
encoder.load_state_dict(torch.load("models/mobilenetv2-e6e8dd43.pth"))


print(encoder)


# ## Building the Decoder - A Multi-Task Lighweight RefineNet
# Paper: https://arxiv.org/pdf/1810.03272.pdf
# ![](https://drsleep.github.io/images/rf_arch.png)


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


num_classes = (40, 1)
decoder = MTLWRefineNet(encoder._out_c, num_classes)
print(decoder)


# # 3 — Train the Model
#
# Now that we've define our encoder and decoder. We are ready to train our model on the NYUDv2 Dataset.
#
# Here's what we'll need:
#
# *   Functions like **train() and valid()**
# *   **An Optimizer and a Loss Function**
# *   **Hyperparameters** such as Weight Decay, Momentum, Learning Rate, Epochs, ...
#
# Doesn't sound so bad, does it?

# ## Loss Function
#
# Let's begin with the Loss and Optimization we'll need.
#
# * The **Segmentation Loss** is the **Cross Entropy Loss**, working as a per-pixel classification function with 15 or so classes.
#
# * The **Depth Loss** will be the **Inverse Huber Loss**.


ignore_index = 255
ignore_depth = 0

crit_segm = nn.CrossEntropyLoss(ignore_index=ignore_index).cuda()
crit_depth = InvHuberLoss(ignore_index=ignore_depth).cuda()


# ## Optimizer
# For the optimizer, we'll use the **Stochastic Gradient Descent**. We'll also add techniques such as weight decay or momentum.


lr_encoder = 1e-2
lr_decoder = 1e-3
momentum_encoder = 0.9
momentum_decoder = 0.9
weight_decay_encoder = 1e-5
weight_decay_decoder = 1e-5


optims = [torch.optim.SGD(encoder.parameters(), lr=lr_encoder, momentum=momentum_encoder, weight_decay=weight_decay_encoder),
          torch.optim.SGD(decoder.parameters(), lr=lr_decoder, momentum=momentum_decoder, weight_decay=weight_decay_decoder)]


# ## Model Definition & State Loading


n_epochs = 1000


init_vals = (0.0, 10000.0)
comp_fns = [operator.gt, operator.lt]
ckpt_dir = "./"
ckpt_path = "./checkpoint.pth.tar"

saver = Saver(
    args=locals(),
    ckpt_dir=ckpt_dir,
    best_val=init_vals,
    condition=comp_fns,
    save_several_mode=all,
)


# .cuda()) # Use .cpu() if you prefer a slow death
hydranet = nn.DataParallel(nn.Sequential(encoder, decoder))

print("Model has {} parameters".format(
    sum([p.numel() for p in hydranet.parameters()])))

start_epoch, _, state_dict = saver.maybe_load(
    ckpt_path=ckpt_path, keys_to_load=["epoch", "best_val", "state_dict"],)
load_state_dict(hydranet, state_dict)

if start_epoch is None:
    start_epoch = 0


print(start_epoch)


# ## Learning Rate Scheduler


opt_scheds = []
for opt in optims:
    opt_scheds.append(torch.optim.lr_scheduler.MultiStepLR(
        opt, np.arange(start_epoch + 1, n_epochs, 100), gamma=0.1))


# ## Training and Validation Loops
#
# Now, all we need to do is go through the Train and Validation DataLoaders, and train our model.


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,), grad_norm=0.0):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input = sample["image"].float().to(device)
        targets = [sample[k].to(device)
                   for k in dataloader.dataset.masks_names]

        # FORWARD
        outputs = model(input)

        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            # TODO: Increment the Loss
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[1:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),)

        # BACKWARD
        for opt in opts:
            opt.zero_grad()
        loss.backward()

        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        # TODO: Run one step
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(
                loss.item(), loss_meter.avg)
        )


def validate(model, metrics, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            # Get the Data
            input = sample["image"].float().to(device)
            targets = [sample[k].to(device)
                       for k in dataloader.dataset.masks_names]

            #input, targets = get_input_and_targets(sample=sample, dataloader=dataloader, device=device)
            targets = [target.squeeze(dim=1).cpu().numpy()
                       for target in targets]

            # Forward
            outputs = model(input)
            #outputs = make_list(outputs)

            # Backward
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(
                        out, size=target.shape[1:], mode="bilinear", align_corners=False)
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals


# ## Main Loop


crop_size = 400
batch_size = 4
val_batch_size = 4
val_every = 5
loss_coeffs = (0.5, 0.5)

for i in range(start_epoch, n_epochs):
    for sched in opt_scheds:
        sched.step(i)

    print("Epoch {:d}".format(i))
    train(hydranet, optims, [crit_segm, crit_depth], trainloader, loss_coeffs)

    if i % val_every == 0:
        metrics = [MeanIoU(num_classes[0]), RMSE(ignore_val=ignore_depth), ]

        with torch.no_grad():
            vals = validate(hydranet, metrics, valloader)
        saver.maybe_save(new_val=vals, dict_to_save={
                         "state_dict": hydranet.state_dict(), "epoch": i})
