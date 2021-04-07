#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import math
import os.path as osp

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset


class CityscapesDataset(Dataset):
    def __init__(self, root=r'/path/to/cityscapes',
                       list_path='./datasets/city_list/train.txt',
                       max_iters=None, transforms=None):
        self.root = root
        self.list_path = list_path
        self.transforms = transforms

        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f:
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])

        if max_iters is not None:
            num_repeats = math.ceil(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * num_repeats
            self.label_ids = self.label_ids * num_repeats

        self.files = []
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]
            img_file = osp.join(self.root, img_name)

            label_file = osp.join(self.root, label_name)

            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafile = self.files[index]

        image = Image.open(datafile['img']).convert('RGB')
        label = Image.open(datafile['label']).convert('L')

        image, label = self.transforms((image, label))

        return image, label, datafile["img_name"]


class DownsampleDataset(Subset):
    def __init__(self, dataset, proportion=0.01, n_samples=None):
        n_samples_dataset = len(dataset)
        if n_samples is None:
            n_samples = int(n_samples_dataset * proportion)
        indices = np.random.randint(n_samples_dataset, size=(n_samples,)).tolist()
        super(DownsampleDataset, self).__init__(dataset, indices)


class CrossCityDataset(Dataset):
    def __init__(self, root, list_path, pseudo_root=None, max_iters=None, transforms=None):
        self.root = root
        self.list_path = list_path
        self.transforms = transforms

        self.img_ids = []
        self.label_ids = []
        with open(list_path) as f:
            for item in f:
                fields = item.strip().split('\t')
                self.img_ids.append(fields[0])
                self.label_ids.append(fields[1])

        if max_iters is not None:
            num_repeats = math.ceil(max_iters / len(self.img_ids))
            self.img_ids = self.img_ids * num_repeats
            self.label_ids = self.label_ids * num_repeats

        self.files = []
        for idx in range(len(self.img_ids)):
            img_name = self.img_ids[idx]
            label_name = self.label_ids[idx]

            img_file = osp.join(self.root, img_name)
            if pseudo_root is None:
                label_file = osp.join(self.root, label_name)
            else:
                label_file = label_name
            self.files.append({
                "img": img_file,
                "label": label_file,
                "img_name": img_name,
                "label_name": label_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafile = self.files[index]

        image = Image.open(datafile['img']).convert('RGB')
        label = Image.open(datafile['label'])  # .convert('L')
        image, label = self.transforms((image, label))
        return image, label, datafile["img_name"]
