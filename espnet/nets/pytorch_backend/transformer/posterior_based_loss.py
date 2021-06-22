#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Copyed from Label smoothing module."""

import torch
from torch import nn

from copy import deepcopy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PosteriorBasedLoss(nn.Module):

    def __init__(
        self,
        size,
        padding_idx,
        smoothing,
        normalize_length=False,
        criterion=nn.KLDivLoss(reduction="none"),
    ):
        #super(LabelSmoothingLoss, self).__init__()
        super(PosteriorBasedLoss, self).__init__()
        self.criterion = criterion
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        self.normalize_length = normalize_length
        self.softmax = torch.nn.Softmax(dim=1)

    # [FIXME] TOO DIRTY
    def padding_for_softloss(self, pred_pad, hyps_list, sentences_scores_list, vocab_size):
        # [ADD] adjusted padding
        pred_max_len = pred_pad.shape[1] # [64, 12, 7807]
        scores_list_max_len = 0
        one_best_list = []
        for i in range(len(sentences_scores_list)):
            scores_list_len = len(sentences_scores_list[i]) # token num exclude first sos of hyps_list
            scores_list_max_len = max(scores_list_max_len, scores_list_len)
            # one-best hyps
            one_best = hyps_list[i][0]['yseq'][1:]
            one_best_list.append(one_best)
        # make zero_pad
        zero_pad = [0] * vocab_size
        zero_pad[0] = 1
        pad_sentences_scores_list = []
        pad_one_best_list = []
        for i, scores_list in enumerate(sentences_scores_list):
            # scores_list : one sentence
            pad_one_sentence_scores_list = []
            pad_one_best = one_best_list[i]
            tokens_num = len(scores_list)
            if tokens_num <= pred_max_len:
                # zero_pad_list padding
                pad_one_sentence_scores_list = scores_list
                # softmax
                pad_one_sentence_scores_list = self.softmax(torch.tensor(pad_one_sentence_scores_list))
                pad_one_sentence_scores_list = pad_one_sentence_scores_list.tolist()
                for j in range(pred_max_len - tokens_num):
                    pad_one_sentence_scores_list.append(zero_pad)
                    # padding
                    # one_best
                    pad_one_best.append(-1)
            elif tokens_num > pred_max_len:
                # reduce soft target tokens
                # [FIXME] unefficient
                pad_one_sentence_scores_list = scores_list[:pred_max_len]
                # softmax
                pad_one_sentence_scores_list = self.softmax(torch.tensor(pad_one_sentence_scores_list))
                pad_one_sentence_scores_list = pad_one_sentence_scores_list.tolist()
                # pad_one_best = pad_one_best[:pred_max_len]
                pad_one_best = one_best_list[i][:pred_max_len]
            pad_sentences_scores_list.append(pad_one_sentence_scores_list)
            pad_one_best_list.append(pad_one_best)
        # pad_sentences_scores_list = torch.tensor(pad_sentences_scores_list)
        # to device
        pad_sentences_scores_list = torch.tensor(pad_sentences_scores_list).to(device)
        pad_one_best_list = torch.tensor(pad_one_best_list).to(device)
        # one_best
        return pad_one_best_list, pad_sentences_scores_list

#            for j in range(pred_max_len): # make pred_max_len
#                if j < len(scores_list):
#                    # [FIXME] uneffective
#                    pad_one_sentence_scores_list.append(scores_list[j])
#                else:
#                    pad_one_sentence_scores_list.append(zero_pad)
#            pad_sentences_scores_list.append(pad_one_sentence_scores_list)
#
#        if pred_max_len >= scores_list_max_len:
#            print(f'smaller: {pred_max_len} {scores_list_max_len}')
#            # [COULDNT]
#            # pad_sentences_scores_list = deepcopy(sentences_scores_list)
#            pad_sentences_scores_list = []
#
#            for i, scores_list in enumerate(sentences_scores_list):
#                # scores_list : one sentence
#                pad_one_sentence_scores_list = []
#                for j in range(pred_max_len): # make pred_max_len
#                    if j < len(scores_list):
#                        # [FIXME] uneffective
#                        pad_one_sentence_scores_list.append(scores_list[j])
#                    else:
#                        pad_one_sentence_scores_list.append(zero_pad)
#                pad_sentences_scores_list.append(pad_one_sentence_scores_list)
#        [FIXME]
#        elif pred_max_len < scores_list_max_len:
#            print(f'bigger: {pred_max_len} {scores_list_max_len}')
#            print('reduce long soft target part')
#            pad_sentences_scores_list = []
#            max_len = pred_max_len
#
#            for i, scores_list in enumerate(sentences_scores_list):
#                # scores_list : one sentence
#                pad_one_sentence_scores_list = []
#                # [FIX]
#                if len(scores_list) > pred_max_len:
#                    for j in range(scores)list)
#                elif len(scores_list) <= pred_max_len:
#
#                for j in range(pred_max_len): # make pred_max_len
#                    if j < len(scores_list):
#                        # [FIXME] uneffective
#                        pad_one_sentence_scores_list.append(scores_list[j])
#                    else:
#                        pad_one_sentence_scores_list.append(zero_pad)
#                pad_sentences_scores_list.append(pad_one_sentence_scores_list)

    def forward(self, x, target, hyps_list, sentences_scores_list, soft_tgt_weight):
        """Compute loss between x and target.

        :param torch.Tensor x: prediction (batch, seqlen, class) (64, 12, 15210)
        :param torch.Tensor target:
            target signal masked with self.padding_id (batch, seqlen)
        :return: scalar float value
        :rtype torch.Tensor
        """
        assert x.size(2) == self.size
        vocab_size = x.size(2)
        pad_one_best_list, soft_target = self.padding_for_softloss(x, hyps_list, sentences_scores_list, vocab_size)

        batch_size = x.size(0)
        x = x.view(-1, self.size)
        target = target.view(-1)
        pad_one_best_list = pad_one_best_list.view(-1)
        soft_target = soft_target.view(-1, self.size)
        # x : [64, 16, 15211] -> [1024, 15211]
        # target : [64, 16] -> [1024]
        if soft_tgt_weight == 0:
           # hard tgt
            with torch.no_grad():
                true_dist = x.clone()
                true_dist.fill_(self.smoothing / (self.size - 1))
                ignore = target == self.padding_idx  # (B,)
                total = len(target) - ignore.sum().item()
                target = target.masked_fill(ignore, 0)  # avoid -1 index
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
            denom = total if self.normalize_length else batch_size
            loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
            return loss

        elif soft_tgt_weight == 1:
            # soft tgt
            # kl = self.criterion(torch.log_softmax(x, dim=1), torch.log_softmax(soft_target, dim=1))
            # soft_target_softmax = []
            # for soft in soft_target:
            #     soft_target_softmax.append(torch.softmax(soft_target_softmax))
            with torch.no_grad():
                ignore = pad_one_best_list == self.padding_idx
            kl = self.criterion(torch.log_softmax(x, dim=1), soft_target)
            denom = total if self.normalize_length else batch_size
            loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
            return loss

        # elif soft_tgt_weight > 0:
        else:
            with torch.no_grad():
                true_dist = x.clone()
                true_dist.fill_(self.smoothing / (self.size - 1))
                ignore_hard = target == self.padding_idx  # (B,)
                total = len(target) - ignore_hard.sum().item()
                target = target.masked_fill(ignore_hard, 0)  # avoid -1 index
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            kl_hard = self.criterion(torch.log_softmax(x, dim=1), true_dist)

            with torch.no_grad():
                ignore_soft = pad_one_best_list == self.padding_idx
            kl_soft = self.criterion(torch.log_softmax(x, dim=1), soft_target)

        denom = total if self.normalize_length else batch_size
        hard_loss = kl_hard.masked_fill(ignore_hard.unsqueeze(1), 0).sum() / denom
        soft_loss = kl_soft.masked_fill(ignore_soft.unsqueeze(1), 0).sum() / denom

        return (1 - soft_tgt_weight) * hard_loss + soft_tgt_weight * soft_loss
