from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
# Torch dataloader for the NYUDv2 dataset

root_dir = "/home/vicky/Coding/Projects/jeremy/nyud"


class HydraNetDataset(Dataset):

    def __init__(self, data_file, transform=None):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [x.decode("utf-8").strip("\n").split("\t")
                         for x in datalist]
        self.root_dir = root_dir
        self.transform = transform
        self.masks_names = ("segm", "depth")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        # Will output list of nyud/*/00000.png
        abs_paths = [os.path.join(self.root_dir, rpath)
                     for rpath in self.datalist[idx]]
        sample = {}
        sample["image"] = np.array(Image.open(abs_paths[0]))

        for mask_name, mask_path in zip(self.masks_names, abs_paths[1:]):
            mask = np.array(Image.open(mask_path))
            assert len(
                mask.shape) == 2, "Masks must be encoded without colourmap"
            sample[mask_name] = mask

        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
        return sample
