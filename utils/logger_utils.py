#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import contextlib
import logging
import os.path as osp

import numpy as np


@contextlib.contextmanager
def np_print_options(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


def set_logger(output_dir=None, log_file=None, debug=False):
    head = '%(asctime)-15s Host %(message)s'
    logger_level = logging.INFO if not debug else logging.DEBUG
    if all((output_dir, log_file)) and len(log_file) > 0:
        logger = logging.getLogger('crosscityadap')
        log_path = osp.join(output_dir, log_file)
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logger_level)
    else:
        logging.basicConfig(level=logger_level, format=head)
        logger = logging.getLogger('crosscityadap')
    return logger
