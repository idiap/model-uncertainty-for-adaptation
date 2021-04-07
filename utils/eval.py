#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import logging
import time

import numpy as np
import torch


logger = logging.getLogger('crosscityadap')


class ScoreUpdater(object):
    # only IoU are computed. accu, cls_accu, etc are ignored.
    def __init__(self, c_num, x_num):
        self._num_class = c_num
        self._num_sample = x_num

        self._confs = torch.zeros(c_num, c_num)
        self._per_cls_iou = torch.zeros(c_num)
        self._start = time.time()
        self._computed = torch.zeros(self._num_sample)  # one-dimension
        self.i = 0
        np.set_printoptions(precision=2)

    def fast_hist(self, label, pred_label, n):
        k = (label >= 0) & (label < n)
        return torch.bincount(n * label[k].int() + pred_label[k], minlength=n ** 2).view(n, n)

    def per_class_iu(self, hist):
        return hist.diag() / (hist.sum(1) + hist.sum(0) - hist.diag())

    def do_updates(self, conf, computed=True):
        if computed:
            self._computed[self.i] = 1
        self._per_cls_iou = self.per_class_iu(conf)

    def update(self, pred_label, label):
        conf = self.fast_hist(label, pred_label, self._num_class)
        self._confs += conf
        self.do_updates(self._confs, True)
        self.scores()
        self.i = self.i + 1

    def scores(self, i=None):
        x_num = self._num_sample
        ious = np.nan_to_num(self._per_cls_iou.cpu().numpy())
        i = self.i
        if i is not None:
            speed = 1. * self._computed.sum() / (time.time() - self._start)
            logger.info('\nDone {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
            # print('\nDone {}/{} with speed: {:.2f}/s'.format(i + 1, x_num, speed))
        name = ''
        logger.info('{}mean iou: {:.2f}%'.format(name, np.mean(ious) * 100))
        print('{}mean iou: {:.2f}%'.format(name, np.mean(ious) * 100))
        with np.printoptions(precision=2, suppress=True):
            logger.info('{}'.format(ious * 100))
            print('{}'.format(ious * 100))

        return ious
