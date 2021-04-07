#
# SPDX-FileCopyrightText: 2021 Idiap Research Institute
#
# Written by Prabhu Teja <prabhu.teja@idiap.ch>,
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F

# from . import IGNORE_LABEL
IGNORE_LABEL = 255


def loss_calc(pred, label):
    #Basic CE loss
    return F.cross_entropy(pred, label.long(), ignore_index=IGNORE_LABEL)


def reg_loss_calc(pred, label, reg_weights, args):
    ce_loss = F.cross_entropy(pred, label.long(), ignore_index=IGNORE_LABEL)

    mask = label != IGNORE_LABEL
    kld = (-pred.log_softmax(1).mean(1) * mask.float()).sum((1, 2)) / (mask.sum((1, 2)).float() + 1)
    kld = (kld * reg_weights.float()).mean()
    total_loss = ce_loss + (args.mr_weight_kld * kld)

    return total_loss


def self_training_regularized_infomax(pred, label, args):
    ce_loss = 0
    if args.lambda_ce > 0:
        ce_loss = F.cross_entropy(pred, label.long(), ignore_index=IGNORE_LABEL)

    # ce_loss = SelfTrainingCrossEntropy(ignore_index=IGNORE_LABEL, threshold=0.99)(pred, label)
    proba = pred.softmax(1)
    n, c, h, w = pred.size()

    
    #entropy_loss
    entropy = -(proba * torch.log2(proba + 1e-10)).sum() / \
        (n * h * w * torch.log2(torch.tensor(c, dtype=torch.float)))

    #diversity loss
    proba = proba.mean(dim=(0, 2, 3))
    diverse = (proba * torch.log2(proba + 1e-10)).sum()

    # if args.kl_prior_variant == 'uniform':
        ## this is uniform prior
    prior = torch.ones_like(proba).to(proba.device)
    # elif args.kl_prior_variant == 'cs':
    #     ##this is CS Prior
    #     prior = 19 * torch.Tensor([0.3687, 0.0608, 0.2282, 0.0066, 0.0088, 0.0123, 0.0021, 0.0055, 0.1593,
    #                                0.0116, 0.0402, 0.0122, 0.0014, 0.0699, 0.0027, 0.0024, 0.0023, 0.0010,
    #                                0.0041]).to(proba.device)
    # elif args.kl_prior_variant == 'gta':
    #     # This is GTA prior
    #     prior = 19 * torch.Tensor([0.3609, 0.0936, 0.1900, 0.0205, 0.0067, 0.0119, 0.0015, 0.0010, 0.0853,
    #                                0.0245, 0.1527, 0.0043, 0.0003, 0.0283, 0.0134, 0.0040, 0.0007, 0.0003,
    #                                0.0001]).to(proba.device)
    # elif args.kl_prior_variant == 'synthia':
    #     prior = 16 * torch.Tensor([0.2021, 0.1966, 0.2999, 0.0030, 0.0029, 0.0103, 0.0004, 0.0010, 0.1051,
    #                                0.0706, 0.0430, 0.0045, 0.0407, 0.0157, 0.0020, 0.0021]).to(proba.device)
    diverse = torch.nn.KLDivLoss(reduction='sum')(prior.log(), proba) * 1.442695
    
    return args.lambda_ce * ce_loss + args.lambda_ent * entropy + args.lambda_div * diverse
    


def self_training_regularized_infomax_rotation_pred(pred, pseudo_label, ssl_pred, ssl_label, args):
    main_part_loss = self_training_regularized_infomax(pred, pseudo_label, args)

    ssl_part_loss = F.cross_entropy(ssl_pred, ssl_label)

    return main_part_loss + args.lambda_unc * ssl_part_loss


class SelfTrainingCrossEntropy(torch.nn.Module):
    def __init__(self, threshold=0.90, ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.loss_scale = torch.tensor([0.99471986, 0.719607, 0.9954222, 0.70447516, 0.69144464,
                                        0.9373348, 0.9491925, 0.8737855, 0.9876312, 0.8852682,
                                        0.9995382, 0.96368825, 0.76953304, 0.9972165, 0.9109088,
                                        0.885779, 0.64849615, 0.884135, 0.76383626]).cuda()

    def forward(self, pred, target=None):
        pred = F.interpolate(pred, size=target.shape[1:], mode='bilinear', align_corners=True)
        
        proba = pred.softmax(1) / self.loss_scale[None, :, None, None]
        prob, target = proba.max(1)
        
        mask = prob < self.threshold
        target[mask] = self.ignore_index
        n_classes = pred.size(1)
        target_mask = (target >= n_classes) * (target == self.ignore_index)
        target[target_mask] = self.ignore_index
        return F.cross_entropy(pred, target, ignore_index=self.ignore_index)



def reg_loss_calc_expand(pred, label, reg_weights, args):
    ce_loss = F.cross_entropy(pred, label.long(), ignore_index=IGNORE_LABEL)

    mask = label != IGNORE_LABEL
    kld = (-pred.log_softmax(1).mean(1) *
           mask.float()).sum((1, 2)) / (mask.sum((1, 2)).float() + 1)
    kld = (kld * reg_weights.float()).mean()

    return ce_loss + (args.mr_weight_kld * kld)


def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size(), f"Shapes are {inputs.size()} and {targets.size()}"
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0:
            loss_mat = torch.Tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean')


def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    inputs = inputs.softmax(dim=1)
    if use_softmax:
        targets = targets.log_softmax(dim=1)

    return F.kl_div(targets, inputs)  ## I'm using reverse KL. 
    

def self_training_regularized_infomax_cct(pred, pseudo_label, aux_pred, args):
    main_part_loss = self_training_regularized_infomax(pred, pseudo_label, args)

    ssl_part_loss = sum([softmax_mse_loss(x, pred.detach(), use_softmax=True) for x in aux_pred])
    # print(main_part_loss.item(), ssl_part_loss.item())
    return main_part_loss + args.lambda_unc * ssl_part_loss
