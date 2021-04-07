#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT


import numpy as np

from . import transforms


def label_mapper():
    mapper =  {7: 0,
            8: 1,
            11: 2,
            12: 255,
            13: 255,
            17: 255,
            19: 3,
            20: 4,
            21: 5,
            22: 255,
            23: 6,
            24: 7,
            25: 8,
            26: 9,
            27: 255,
            28: 10,
            31: 255,
            32: 11,
            33: 12}
    arr = 255 * np.ones((255, ))
    for x in mapper:
        arr[x] = mapper[x]
    return arr


def get_train_transforms(args, mine_id):
    label_to_id = label_mapper()

    train_src_transforms = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomScaleCrop(args.base_size, args.input_size, args.train_scale_src, [], 0.0),
        transforms.DefaultTransforms(),
        transforms.RemapLabels(label_to_id)
    ]

    train_tgt_transforms = [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomScaleCrop(args.base_size, args.input_size,
                                       args.train_scale_tgt, [], []),
        transforms.DefaultTransforms(),
        # transforms.RemapLabels(label_to_id)
    ]
    train_src_transforms = transforms.Compose(train_src_transforms)
    train_tgt_transforms = transforms.Compose(train_tgt_transforms)
    return train_src_transforms, train_tgt_transforms


def get_test_transforms():
    label_to_id = label_mapper()
    tgt_transforms = [
        transforms.Resize((1024, 512)), 
        transforms.DefaultTransforms(),
        transforms.RemapLabels(label_to_id)
    ]
    return transforms.Compose(tgt_transforms)


def get_val_transforms(args):
    label_to_id = label_mapper()
    tgt_transforms = [
        transforms.Resize((1024, 512)),
        transforms.DefaultTransforms(),
        # transforms.RemapLabels(label_to_id)  ## Getting rid of this because NTHU has no labels for train set. 
    ]
    return transforms.Compose(tgt_transforms)
