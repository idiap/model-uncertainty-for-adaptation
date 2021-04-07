#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, tot_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, tot_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if not args.freeze_classifier:
        optimizer.param_groups[1]['lr'] = lr * 10
