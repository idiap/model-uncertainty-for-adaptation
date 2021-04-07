#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torchvision import transforms

Compose = transforms.Compose


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, mask = sample
        return TF.resize(img, self.size[::-1], Image.BICUBIC), TF.resize(mask, self.size[::-1], Image.NEAREST)


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __call__(self, sample):
        img, mask = sample
        img = (img - self.mean) / self.std
        return img, mask


class ToTensor:
    """Convert ndarrays is not None to Tensors."""

    def __call__(self, sample):
        img, mask = sample
        img = TF.to_tensor(img)
        mask = torch.from_numpy(np.array(mask, np.float32)).float()
        return img, mask


class SetSeed:
    def __call__(self, sample):
        if 'seed' in sample:
            random.seed(sample['seed'])
        return sample


class RandomHorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, mask = sample

        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask


class __RandomRotate:
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, mask = sample

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class RandomScaleCrop:
    def __init__(self, base_size, crop_size, scale, rare_ids, mine_probability):
        self.base_size = base_size * (3 - len(base_size))
        self.crop_size = crop_size * (3 - len(crop_size))
        self.scale = scale
        self.rare_ids = rare_ids
        self.mine_prob = mine_probability
        self.fill = 0

    def __call__(self, sample):
        img, mask = sample

        # random scale (short edge)
        w, h = img.size
        if h > w:
            short_size = random.randint(int(self.base_size[0] * self.scale[0]), int(self.base_size[0] * self.scale[1]))
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            short_size = random.randint(int(self.base_size[1] * self.scale[0]), int(self.base_size[1] * self.scale[1]))
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        padh = self.crop_size[1] - oh if oh < self.crop_size[1] else 0
        padw = self.crop_size[0] - ow if ow < self.crop_size[0] else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        
        rare_label_present = np.intersect1d(np.array(mask), self.rare_ids)
        if (rare_label_present.size > 0) and random.random() < self.mine_prob:
            chosen_rand_id = np.random.choice(rare_label_present)
            print("Rare ids present: ", rare_label_present, "Random class chosen: ", chosen_rand_id)
            all_loc_x, all_loc_y = np.where(np.array(mask) == chosen_rand_id)
            # rand_ind = np.random.choice(all_loc_x.size)
            # center = [all_loc_x[rand_ind], all_loc_y[rand_ind]]
            center = [int(all_loc_x.mean()), int(all_loc_y.mean())]
            print(center)
        else:
            center = None
        
        return self.random_crop(img, mask, crop_center=center)
        
    
    def random_crop(self, img, mask, crop_center=None):
        w, h = img.size
        
        # random crop crop_size
        if crop_center is None:
            ## basically select the mid point (first time below) in (crop_size to  w-cropsize).
            ## then get the lowest point mid point by subtracting the cropsize/2
            x1 = random.randint(0, w - self.crop_size[0])
            y1 = random.randint(0, h - self.crop_size[1])
            img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
            mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
            # print("line 141:", w, h, self.crop_size, img.size)
        else:
            x1 = crop_center[0] - self.crop_size[0] // 2
            y1 = crop_center[1] - self.crop_size[1] // 2
    
            img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
            mask = mask.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))

        return img, mask


class CustomTranslate:
    def __init__(self, translations, fill=(0, 0, 0)):
        self.translations = translations
        self.fill = fill

    def __call__(self, img1):
        translations = []
        for translate in self.translations:
            im_tr = TF.affine(img1, translate=translate, angle=0,
                              scale=1, shear=0, fillcolor=self.fill)
            translations.append(im_tr)
        return translations


class DefaultTransforms: 
    def __call__(self, sample):
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        image, label = sample
        image = np.asarray(image, np.float32).copy()
        image = image[:, :, ::-1]  # change to BGR
        image -= IMG_MEAN
        image = image.transpose((2, 0, 1)).copy()  # (C x H x W)
        new_image = torch.from_numpy(image)  # (C x H x W)

        return new_image, torch.from_numpy(np.array(label).copy()).float()


class RemapLabels:
    def __init__(self, label_map):
        self.label_map = label_map

    def __call__(self, sample):
        if isinstance(self.label_map, dict):
            return sample[0], np.vectorize(self.label_map.get)(sample[1])  #.astype(np.int64)

        if torch.is_tensor(sample[1]):
            remapped = self.label_map[sample[1].int().numpy()]
        remapped = torch.from_numpy(remapped).float()
        return sample[0], remapped


class OnehotLabels:
    def __init__(self, n_classes=19):
        self.n_classes = n_classes

    def __call__(self, sample):
        img, label = sample
        l_shape = label.shape
        
        y = label.ravel()
        y_one_hot = np.zeros((y.shape[0], self.n_classes))
        y_one_hot[np.arange(y.shape[0], y)] = 1

        return img, y_one_hot.reshape(l_shape + (self.n_classes, )).transpose(2, 0, 1)


