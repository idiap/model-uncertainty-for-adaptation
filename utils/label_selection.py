#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import logging
import os
import shutil
import time

import numpy as np
from PIL import Image

from .viz_utils import colorize_mask

logger = logging.getLogger('crosscityadap')
osp = os.path


def kc_parameters(conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args):
    logger.info('###### Start kc generation in round {} ! ######'.format(round_idx))
    start_kc = time.time()
    # threshold for each class
    conf_tot = 0.0
    cls_thresh = np.ones(args.num_classes, dtype=np.float32)
    cls_sel_size = np.zeros(args.num_classes, dtype=np.float32)
    cls_size = np.zeros(args.num_classes, dtype=np.float32)

    if (args.kc_policy == 'cb') and (args.kc_value == 'conf'):
        print()
        for idx_cls in np.arange(0, args.num_classes):
            cls_size[idx_cls] = pred_cls_num[idx_cls]
            if conf_dict[idx_cls] != None:
                conf_dict[idx_cls].sort(reverse=True)  # sort in descending order
                len_cls = len(conf_dict[idx_cls])
                cls_sel_size[idx_cls] = int(len_cls * tgt_portion)
                len_cls_thresh = int(cls_sel_size[idx_cls])
                if len_cls_thresh != 0:
                    cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
                conf_dict[idx_cls] = None
    logger.info("Per class thresholds:")
    logger.info(cls_thresh)
    # threshold for mine_id with priority
    num_mine_id = len(np.nonzero(cls_size / np.sum(cls_size) < 0.001)[0])
    # chose the smallest mine_id
    id_all = np.argsort(cls_size / np.sum(cls_size))
    rare_id = id_all[:3]
    mine_id = id_all[:num_mine_id]  # sort mine_id in ascending order w.r.t predication portions
    # save mine ids
    np.save(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy', rare_id)
    np.save(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy', mine_id)
    logger.info('Mining ids : {}! {} rarest ids: {}!'.format(mine_id, 3, rare_id))
    # save thresholds
    np.save(save_stats_path + '/cls_thresh_round' + str(round_idx) + '.npy', cls_thresh)
    np.save(save_stats_path + '/cls_sel_size_round' + str(round_idx) + '.npy', cls_sel_size)
    logger.info('###### Finish kc generation in round {}! Time cost: {:.2f} seconds. ######'.format(
        round_idx, time.time() - start_kc))
    return cls_thresh


def label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path, save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args):
    logger.info('###### Start pseudo-label generation in round {} ! ######'.format(round_idx))
    start_pl = time.time()
    filenames = [osp.splitext(x)[0] for x in os.listdir(save_prob_path) if x.endswith('npy')]

    for sample_name in filenames:
        probmap_path = osp.join(save_prob_path, '{}.npy'.format(sample_name))
        pred_path = osp.join(save_pred_path, '{}.png'.format(sample_name))

        pred_prob = np.load(probmap_path)
        pred_label_labelIDs = np.asarray(Image.open(pred_path))

        if args.kc_policy == 'cb':
            save_wpred_vis_path = osp.join(save_round_eval_path, 'weighted_pred_vis')
            if not os.path.exists(save_wpred_vis_path):
                os.makedirs(save_wpred_vis_path)
            weighted_prob = pred_prob / cls_thresh
            weighted_prob_ids = weighted_prob.argmax(axis=2).astype(np.uint8)

            if args.debug:
                colorize_mask(weighted_prob_ids).save('%s/%s_color.png' % (save_wpred_vis_path, sample_name))

            weighted_conf = weighted_prob.max(axis=2)
            pred_label_labelIDs = weighted_prob_ids
            pred_label_labelIDs[weighted_conf < 1] = 255  # '255' in cityscapes indicates 'unlabaled' for trainIDs

        # save colored pseudo-label map
        if args.debug:
            pseudo_label_col = colorize_mask(pred_label_labelIDs)
            pseudo_label_col.save('%s/%s_color.png' % (save_pseudo_label_color_path, sample_name))
        # save pseudo-label map with label IDs
        pseudo_label_save = Image.fromarray(pred_label_labelIDs.astype(np.uint8))
        pseudo_label_save.save('%s/%s.png' % (save_pseudo_label_path, sample_name))
    # remove probability maps

    shutil.rmtree(save_prob_path)

    logger.info('###### Finish pseudo-label generation in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time() - start_pl))
