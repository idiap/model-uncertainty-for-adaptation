#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import csv
import os.path as osp
import random

import numpy as np


def parse_split_list(list_name):
    image_list, image_name_list, label_list = [], [], []

    with open(list_name) as fp:
        for item in fp:
            image, label = item.strip().split('\t')
            image_name = image.split('/')[-1]

            image_list.append(image)
            image_name_list.append(image_name)
            label_list.append(label)

    return image_list, image_name_list, label_list, len(image_list)


def savelst_srctgt(src_portion, image_tgt_list, image_name_tgt_list, image_src_list, label_src_list,
                   save_lst_path, save_pseudo_label_path, src_num, tgt_num, randseed, args):

    np.random.seed(randseed)

    src_num_sel = int(np.floor(src_num*src_portion))

    sel_src_img_list = random.sample(image_src_list, src_num_sel)
    sel_src_label_list = random.sample(label_src_list, src_num_sel)

    src_train_lst = osp.join(save_lst_path, 'src_train.lst')
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.lst')

    # generate src train list
    with open(src_train_lst, 'w') as fp:
        tsv_writer = csv.writer(fp, delimiter='\t')
        for row in zip(sel_src_img_list, sel_src_label_list):
            tsv_writer.writerow(row)

    # generate tgt train list
    with open(tgt_train_lst, 'w') as fp:
        tsv_writer = csv.writer(fp, delimiter='\t')
        if args.lr_weight_ent > 0:
            for x, y in zip(image_tgt_list, image_name_tgt_list):
                soft_label_tgt_path = osp.join(save_pseudo_label_path, y.split('.')[0] + '.npy')
                image_tgt_path = osp.join(save_pseudo_label_path, y)
                tsv_writer.writerow([x, image_tgt_path, soft_label_tgt_path])

        elif args.lr_weight_ent == 0:
            for x, y in zip(image_tgt_list, image_name_tgt_list):
                y = y.replace('jpg', 'png')
                tsv_writer.writerow([x, osp.join(save_pseudo_label_path, y)])

    return src_train_lst, tgt_train_lst, src_num_sel


def savelst_tgt(image_tgt_list, image_name_tgt_list, save_lst_path, save_pseudo_label_path):
    tgt_train_lst = osp.join(save_lst_path, 'tgt_train.lst')
    # generate tgt train list
    with open(tgt_train_lst, 'w') as f:
        for x, y in zip(image_tgt_list, image_name_tgt_list):
            y = y.replace('jpg', 'png')
            image_tgt_path = osp.join(save_pseudo_label_path, y)
            f.write("%s\t%s\n" % (x, image_tgt_path))

    return tgt_train_lst
