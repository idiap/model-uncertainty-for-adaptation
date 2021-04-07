#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import os
import random
import shutil

import numpy as np
import torch


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    #torch.backends.cudnn.deterministic = True


def cleanup(directory):
    savedir = directory
    to_delete = [x for x in os.listdir(savedir) if x.isnumeric()]
    for folder in to_delete:
        full_path = os.path.join(savedir, folder)
        shutil.rmtree(full_path)
