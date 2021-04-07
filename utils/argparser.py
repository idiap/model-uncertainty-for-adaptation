#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT


import argparse


DATA_TGT_DIRECTORY = '/path/to/cross/city/folder'
DATA_TGT_TRAIN_LIST_PATH = './datasets/NTHU_list/{}/train.txt'
DATA_TGT_TEST_LIST_PATH = './datasets/NTHU_list/{}/test.txt'
### train ###
BATCH_SIZE = 2
RANDSEED = 3
# params for optimizor
LEARNING_RATE = 5e-5
POWER = 0.0
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
EPR = 2
SRC_SAMPLING_POLICY = 'r'
KC_POLICY = 'cb'
KC_VALUE = 'conf'
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
# varies but dataset
MAX_SRC_PORT = 1  # 0.06;
SRC_PORT_STEP = 0  # 0.0025:
MINE_PORT = 1e-3
RARE_CLS_NUM = 3
MINE_CHANCE = 0.8
### val ###
SAVE_PATH = 'debug'
TEST_IMAGE_SIZE = (512, 1024)[::-1]
EVAL_SCALE = 0.9
TEST_SCALE = (0.9, 1.0, 1.2)
DS_RATE = 10


def common_args():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network", conflict_handler='resolve')
    parser.add_argument("--data-tgt-dir", type=str, default=DATA_TGT_DIRECTORY,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-tgt-train-list", type=str, default=DATA_TGT_TRAIN_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target train dataset.")
    parser.add_argument("--data-tgt-test-list", type=str, default=DATA_TGT_TEST_LIST_PATH,
                        help="Path to the file listing the images*GT labels in the target test dataset.")
    parser.add_argument('--debug', help='True means logging debug info.',
                        default=False, action='store_true')
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")

    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    ### val

    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument('--kc-policy', default=KC_POLICY, type=str, choices=['kc', 'none'],
                        help='The policy to determine kc. "cb" for weighted class-balanced threshold')
    parser.add_argument('--kc-value', default=KC_VALUE, type=str, choices=['conf', 'prob'],
                        help='The way to determine kc values, either "conf", or "prob".')
    parser.add_argument('--ds-rate', default=DS_RATE, type=int,
                        help='The downsampling rate in kc calculation.')
    parser.add_argument('--init-tgt-port', default=INIT_TGT_PORT, type=float, dest='init_tgt_port',
                        help='The initial portion of target to determine kc')
    parser.add_argument('--max-tgt-port', default=MAX_TGT_PORT, type=float, dest='max_tgt_port',
                        help='The max portion of target to determine kc')
    parser.add_argument('--tgt-port-step', default=TGT_PORT_STEP, type=float, dest='tgt_port_step',
                        help='The portion step in target domain in every round of self-paced self-trained neural network')

    parser.add_argument('--src-port-step', default=SRC_PORT_STEP, type=float, dest='src_port_step',
                        help='The portion step in source domain in every round of self-paced self-trained neural network')
    parser.add_argument('--randseed', default=RANDSEED, type=int,
                        help='The random seed to sample the source dataset.')

    parser.add_argument('--no-src-data', action='store_true', 
                        help='Flag to not use source data for adaptation. This work requires this flag appended to each run')
    parser.add_argument('--info-max-loss', action='store_true',
                        help='Use the infomation maximization loss')
    parser.add_argument('--lambda-ent', default=1, type=float, help='Reg weight for entropy')
    parser.add_argument('--lambda-div', default=0.0, type=float, help='Reg weight for diversity (Dont use this. Hurts perf)')
    parser.add_argument('--lambda-ce', default=1, type=float, help='Reg weight for CE for pseudo-labels')
    parser.add_argument('--lambda-unc', default=0.1, type=float, help='Reg weight for Uncertainity loss')
    parser.add_argument('--freeze-classifier', action='store_true', help='Freeze ASPP classifiers')
    parser.add_argument('--unc-noise', action='store_true', help='Use Dropout based uncertainty noise')
    parser.add_argument('--decoders', default=4, type=int, help='Number of auxiliary decoders')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout proba of auxiliary decoders')
    parser.add_argument('-f', '--file', help='quick hack for jupyter')

    return parser


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
        A list of parsed arguments.
    """

    NUM_ROUNDS = 1
    INPUT_SIZE = (512, 1024)[::-1]  
    # train scales for src and tgt
    TRAIN_SCALE_SRC = (0.5, 1.5)
    TRAIN_SCALE_TGT = (0.5, 1.5)

    
    # DATA_SRC_LIST_PATH = './dataset/list/gta5/train_small.lst'

    RESTORE_FROM = './pretrained/Cityscapes_source_class13.pth'
    NUM_CLASSES = 13
    INIT_SRC_PORT = 1
    BASE_SIZE = (1024, 512)

    parser = common_args()
    parser.add_argument('--city', type=str, choices=['Rio', 'Rome', 'Taipei', 'Tokyo'])
    parser.add_argument('--base-size', type=int, nargs=2, default=BASE_SIZE)
    
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--input-size", type=int, nargs=2, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")

    parser.add_argument("--train-scale-src", type=int, nargs=2, default=TRAIN_SCALE_SRC,
                        help="The scale for multi-scale training in source domain.")
    parser.add_argument("--train-scale-tgt", type=int, nargs=2, default=TRAIN_SCALE_TGT,
                        help="The scale for multi-scale training in target domain.")
    parser.add_argument("--num-rounds", type=int, default=NUM_ROUNDS,
                        help="Number of rounds for self-training.")
    parser.add_argument("--epr", type=int, default=EPR,
                        help="Number of epochs per round for self-training.")

    parser.add_argument('--init-src-port', default=INIT_SRC_PORT, type=float, dest='init_src_port',
                        help='The initial portion of source portion for self-trained neural network')
    parser.add_argument('--max-src-port', default=MAX_SRC_PORT, type=float, dest='max_src_port',
                        help='The max portion of source portion for self-trained neural network')

    return parser.parse_known_args()[0]


