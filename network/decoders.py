#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>
#
# SPDX-License-Identifier: MIT


import torch.nn.functional as F
from torch import nn

from .models.deeplab_multi import Classifier_Module


def upsample(num_classes=13):
    return Classifier_Module(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)


class DropOutDecoder(nn.Module):
    def __init__(self, drop_rate=0.3, decoder=None):
        super().__init__()
        self.dropout = nn.Dropout2d(p=drop_rate)
        self.upsample = upsample() if decoder is None else decoder

    def forward(self, x, *ig, **ign):
        x = self.upsample(self.dropout(x))
        return x


class JointSegAuxDecoderModel(nn.Module):
    def __init__(self, seg_model, auxmodule):
        super().__init__()
        self.seg_model = seg_model
        self.aux_decoders = auxmodule

    def forward(self, x, training=False):
        seg_pred, features = self.seg_model(x, return_features=True)
        if not training:
            return seg_pred
        perturbed_out = self.aux_decoders(features)
        input_size = x.size()[2:]
        perturbed_out = [self.interp(x, input_size) for x in perturbed_out]
    
        return seg_pred, perturbed_out

    def optim_parameters(self, args):
        return self.seg_model.optim_parameters(args) + self.aux_decoders.optim_parameters(args)

    def interp(self, x1, input_size):
        return F.interpolate(x1, size=input_size, mode='bilinear', align_corners=True)


class NoisyDecoders(nn.Module):
    def __init__(self, n_decoders, dropout):
        super().__init__()
        self.decoders = nn.ModuleList([DropOutDecoder(drop_rate=dropout) for _ in range(n_decoders)])

    def forward(self, x):
        return [decoder(x) for decoder in self.decoders]

    def optim_parameters(self, args):
        return [{'params': self.parameters(), 'lr': 10 * args.learning_rate}]
