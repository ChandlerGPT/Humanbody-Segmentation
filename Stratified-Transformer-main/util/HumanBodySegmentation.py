import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import collate_fn, collate_fn_limit
from util.data_util import data_prepare_humanbody as data_prepare
import random

class HumanBodySegmentation(Dataset):
    def __init__(self, split='train', data_root='Data/HumanBodySegmentationDataset/', voxel_size=0.04, voxel_max=80000, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        self.features_list = sorted(glob.glob(os.path.join(data_root, split, "*_features.npy")))

        self.labels_list = [x.replace("_features.npy", "_labels.npy") for x in self.features_list]

        print(f"HumanBodySegmentation {split} set: {len(self.features_list)} samples found.")

    def __getitem__(self, idx):
        data_idx = idx % len(self.features_list)
        feature_path = self.features_list[data_idx]
        label_path = self.labels_list[data_idx]

        # 加载点云坐标与标签
        coord = np.load(feature_path).astype(np.float32)  # (N,3)
        label = np.load(label_path).astype(np.int32)      # (N,)

        feat = np.zeros((coord.shape[0], 3), dtype=np.float32)

        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)

        return coord, feat, label

    def __len__(self):
        return len(self.features_list) * self.loop

def worker_init_fn(worker_id):
    random.seed(123 + worker_id)

if __name__ == "__main__":
    dataset = HumanBodySegmentation(split='train', data_root='Data/HumanBodySegmentationDataset/', voxel_size=0.04, voxel_max=80000)
    print("Dataset length:", len(dataset))
    coord, feat, label = dataset[0]
    print("coord shape:", coord.shape, "feat shape:", feat.shape, "label shape:", label.shape)
