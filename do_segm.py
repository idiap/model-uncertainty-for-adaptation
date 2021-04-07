#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT


import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import (CityscapesDataset, CrossCityDataset, get_test_transforms,
                      get_train_transforms)
from generate_pseudo_labels import validate_model
from network import DeeplabMulti as DeepLab
from network import JointSegAuxDecoderModel, NoisyDecoders
from utils import (ScoreUpdater, adjust_learning_rate, cleanup,
                   get_arguments, label_selection, parse_split_list,
                   savelst_tgt, seed_torch, self_training_regularized_infomax,
                   self_training_regularized_infomax_cct, set_logger)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
osp = os.path


args = get_arguments()
if not os.path.exists(args.save):
    os.makedirs(args.save)
logger = set_logger(args.save, 'training_logger', False)


def make_network(args):
    model = DeepLab(13, False)
    model = torch.nn.DataParallel(model)
    sd = torch.load('pretrained/Cityscapes_source_class13.pth', map_location=device)['state_dict']
    model.load_state_dict(sd)

    model = model.module
    if args.unc_noise:
        aux_decoders = NoisyDecoders(args.decoders, args.dropout)
        model = JointSegAuxDecoderModel(model, aux_decoders)
    return model


def test(model, round_idx):
    transforms = get_test_transforms()
    
    ds = CrossCityDataset(root=args.data_tgt_dir, list_path=args.data_tgt_test_list.format(args.city), transforms=transforms)
    loader = torch.utils.data.DataLoader(ds, batch_size=6, pin_memory=torch.cuda.is_available(), num_workers=6)

    scorer = ScoreUpdater(13, len(loader))
    logger.info('###### Start evaluating in target domain test set in round {}! ######'.format(round_idx))
    start_eval = time.time()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img, label, _ = batch
            pred = model(img.to(device)).argmax(1).cpu()
            scorer.update(pred.view(-1), label.view(-1))
    model.train()
    logger.info('###### Finish evaluating in target domain test set in round {}! Time cost: {:.2f} seconds. ######'.format(
        round_idx, time.time()-start_eval))


def train(mix_trainloader, model, interp, optimizer, args):
    """Create the model and start the training."""
    tot_iter = len(mix_trainloader)
    for i_iter, batch in enumerate(mix_trainloader):
        images, labels, name = batch
        labels = labels.long()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, tot_iter, args)

        if args.info_max_loss:
            pred = model(images.to(device), training=True)
            loss = self_training_regularized_infomax(pred, labels.to(device), args)
        elif args.unc_noise:
            pred, noise_pred = model(images.to(device), training=True)
            loss = self_training_regularized_infomax_cct(pred, labels.to(device), noise_pred, args)
        else:
            pred = model(images.to(device))
            loss = F.cross_entropy(pred, labels.to(device), ignore_index=255)

        loss.backward()
        optimizer.step()

        logger.info('iter = {} of {} completed, loss = {:.4f}'.format(i_iter+1, tot_iter, loss.item()))


def main():
    seed_torch(args.randseed)

    logger.info('Starting training with arguments')
    logger.info(vars(args))

    save_path = args.save
    save_pseudo_label_path = osp.join(save_path, 'pseudo_label')  # in 'save_path'. Save labelIDs, not trainIDs.
    save_stats_path = osp.join(save_path, 'stats')  # in 'save_path'
    save_lst_path = osp.join(save_path, 'list')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)
    if not os.path.exists(save_lst_path):
        os.makedirs(save_lst_path)

    tgt_portion = args.init_tgt_port
    image_tgt_list, image_name_tgt_list, _, _ = parse_split_list(args.data_tgt_train_list.format(args.city))

    model = make_network(args).to(device)
    test(model, -1)
    for round_idx in range(args.num_rounds):
        save_round_eval_path = osp.join(args.save, str(round_idx))
        save_pseudo_label_color_path = osp.join(
            save_round_eval_path, 'pseudo_label_color')  # in every 'save_round_eval_path'
        if not os.path.exists(save_round_eval_path):
            os.makedirs(save_round_eval_path)
        if not os.path.exists(save_pseudo_label_color_path):
            os.makedirs(save_pseudo_label_color_path)
        src_portion = args.init_src_port
        ########## pseudo-label generation
        conf_dict, pred_cls_num, save_prob_path, save_pred_path = validate_model(model,
                                                                                 save_round_eval_path,
                                                                                 round_idx, args)
        cls_thresh = label_selection.kc_parameters(
            conf_dict, pred_cls_num, tgt_portion, round_idx, save_stats_path, args)

        label_selection.label_selection(cls_thresh, round_idx, save_prob_path, save_pred_path,
                                        save_pseudo_label_path, save_pseudo_label_color_path, save_round_eval_path, args)

        tgt_portion = min(tgt_portion + args.tgt_port_step, args.max_tgt_port)
        tgt_train_lst = savelst_tgt(image_tgt_list, image_name_tgt_list, save_lst_path, save_pseudo_label_path)

        rare_id = np.load(save_stats_path + '/rare_id_round' + str(round_idx) + '.npy')
        mine_id = np.load(save_stats_path + '/mine_id_round' + str(round_idx) + '.npy')
        # mine_chance = args.mine_chance

        src_transforms, tgt_transforms = get_train_transforms(args, mine_id)
        srcds = CityscapesDataset(transforms=src_transforms)

        tgtds = CrossCityDataset(args.data_tgt_dir.format(args.city), tgt_train_lst,
                                  pseudo_root=save_pseudo_label_path, transforms=tgt_transforms)
        
        if args.no_src_data:
            mixtrainset = tgtds
        else:
            mixtrainset = torch.utils.data.ConcatDataset([srcds, tgtds])

        mix_loader = DataLoader(mixtrainset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.batch_size, pin_memory=torch.cuda.is_available())
        src_portion = min(src_portion + args.src_port_step, args.max_src_port)
        optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
        interp = nn.Upsample(size=args.input_size[::-1], mode='bilinear', align_corners=True)
        torch.backends.cudnn.enabled = True  # enable cudnn
        torch.backends.cudnn.benchmark = True
        start = time.time()
        for epoch in range(args.epr):
            train(mix_loader, model, interp, optimizer, args)
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.save,
                                                    '2nthy_round' + str(round_idx) + '_epoch' + str(epoch) + '.pth'))
        end = time.time()
        
        logger.info('###### Finish model retraining dataset in round {}! Time cost: {:.2f} seconds. ######'.format(
            round_idx, end - start))
        test(model, round_idx)
        cleanup(args.save)
    cleanup(args.save)
    shutil.rmtree(save_pseudo_label_path)
    test(model, args.num_rounds - 1)


if __name__ == "__main__":
    main()
