#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math

# [ADD]
from collections import Counter
from pathlib import Path
import pickle

import torch

from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator as ASRErrorCalculator
from espnet.nets.e2e_mt_common import ErrorCalculator as MTErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_st import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
# [ADD]
from espnet.nets.pytorch_backend.transformer.posterior_based_loss import (
    PosteriorBasedLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.st_interface import STInterface
from espnet.utils.fill_missing_args import fill_missing_args


class E2E(STInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument(
            "--transformer-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize transformer parameters",
        )
        group.add_argument(
            "--transformer-input-layer",
            type=str,
            default="conv2d",
            choices=["conv2d", "linear", "embed"],
            help="transformer input layer type",
        )
        group.add_argument(
            "--transformer-attn-dropout-rate",
            default=None,
            type=float,
            help="dropout in transformer attention. use --dropout-rate if None is set",
        )
        group.add_argument(
            "--transformer-lr",
            default=10.0,
            type=float,
            help="Initial value of learning rate",
        )
        group.add_argument(
            "--transformer-warmup-steps",
            default=25000,
            type=int,
            help="optimizer warmup steps",
        )
        group.add_argument(
            "--transformer-length-normalized-loss",
            default=False,
            type=strtobool,
            help="normalize loss by length",
        )
        group.add_argument(
            "--transformer-encoder-selfattn-layer-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
            ],
            help="transformer encoder self-attention layer type",
        )
        group.add_argument(
            "--transformer-decoder-selfattn-layer-type",
            type=str,
            default="selfattn",
            choices=[
                "selfattn",
                "lightconv",
                "lightconv2d",
                "dynamicconv",
                "dynamicconv2d",
                "light-dynamicconv2d",
            ],
            help="transformer decoder self-attention layer type",
        )
        # Lightweight/Dynamic convolution related parameters.
        # See https://arxiv.org/abs/1912.11793v2
        # and https://arxiv.org/abs/1901.10430 for detail of the method.
        # Configurations used in the first paper are in
        # egs/{csj, librispeech}/asr1/conf/tuning/ld_conv/
        group.add_argument(
            "--wshare",
            default=4,
            type=int,
            help="Number of parameter shargin for lightweight convolution",
        )
        group.add_argument(
            "--ldconv-encoder-kernel-length",
            default="21_23_25_27_29_31_33_35_37_39_41_43",
            type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Encoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",
        )
        group.add_argument(
            "--ldconv-decoder-kernel-length",
            default="11_13_15_17_19_21",
            type=str,
            help="kernel size for lightweight/dynamic convolution: "
            'Decoder side. For example, "21_23_25" means kernel length 21 for '
            "First layer, 23 for Second layer and so on.",
        )
        group.add_argument(
            "--ldconv-usebias",
            type=strtobool,
            default=False,
            help="use bias term in lightweight/dynamic convolution",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.1,
            type=float,
            help="Dropout rate for the encoder",
        )
        # Encoder
        group.add_argument(
            "--elayers",
            default=4,
            type=int,
            help="Number of encoder layers",
        )
        group.add_argument(
            "--eunits",
            "-u",
            default=2048,
            type=int,
            help="Number of encoder hidden units",
        )
        # Attention
        group.add_argument(
            "--adim",
            default=256,
            type=int,
            help="Number of attention transformation dimensions",
        )
        group.add_argument(
            "--aheads",
            default=4,
            type=int,
            help="Number of heads for multi head attention",
        )
        # Decoder
        group.add_argument(
            "--dlayers", default=6, type=int, help="Number of decoder layers"
        )
        group.add_argument(
            "--dunits", default=2048, type=int, help="Number of decoder hidden units"
        )
        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_encoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.decoder = Decoder(
            odim=odim,
            selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            conv_wshare=args.wshare,
            conv_kernel_length=args.ldconv_decoder_kernel_length,
            conv_usebias=args.ldconv_usebias,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        self.pad = 0  # use <blank> for padding
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="st", arch="transformer")
        self.reporter = Reporter()
        # [ADD]
        self.batch_size = args.batch_size
        self.criterion_st = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight_st,
            args.transformer_length_normalized_loss,
        )
        # asrpbl
        self.criterion_asr = PosteriorBasedLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight_asr,
            args.transformer_length_normalized_loss,
        )
        self.criterion_mt = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight_mt,
            args.transformer_length_normalized_loss,
        )
        # submodule for ASR task
        self.mtlalpha = args.mtlalpha
        self.asr_weight = args.asr_weight
        if self.asr_weight > 0 and args.mtlalpha < 1:
            self.decoder_asr = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
            )

        # submodule for MT task
        self.mt_weight = args.mt_weight
        if self.mt_weight > 0:
            self.encoder_mt = Encoder(
                idim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                input_layer="embed",
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                padding_idx=0,
            )
        self.reset_parameters(args)  # NOTE: place after the submodule initialization
        self.adim = args.adim  # used for CTC (equal to d_model)
        if self.asr_weight > 0 and args.mtlalpha > 0.0:
            self.ctc = CTC(
                odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True
            )
        else:
            self.ctc = None

        # translation error calculator
        self.error_calculator = MTErrorCalculator(
            args.char_list, args.sym_space, args.sym_blank, args.report_bleu
        )
        # recognition error calculator
        self.error_calculator_asr = ASRErrorCalculator(
            args.char_list,
            args.sym_space,
            args.sym_blank,
            args.report_cer,
            args.report_wer,
        )
        self.rnnlm = None

        # multilingual E2E-ST related
        self.multilingual = getattr(args, "multilingual", False)
        self.replace_sos = getattr(args, "replace_sos", False)
        # [ADD]
        self.args = args

    def reset_parameters(self, args):
        """Initialize parameters."""
        initialize(self, args.transformer_init)
        if self.mt_weight > 0:
            torch.nn.init.normal_(
                self.encoder_mt.embed[0].weight, mean=0, std=args.adim ** -0.5
            )
            torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)

    # [WILL ADD -> ADD] batch more information : uttid
    # [ADD] uttid_list
    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src, uttid_list):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_pad_src: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 0. Extract target language ID
        tgt_lang_ids = None
        if self.multilingual:
            tgt_lang_ids = ys_pad[:, 0:1]
            ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining

        # 1. forward encoder
        # this is mel batch
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        self.xs_pad = xs_pad
        '''
        # not pre-decoding version
        from torch.nn.utils.rnn import pack_padded_sequence
        from espnet.asr.pytorch_backend.asr_init import load_trained_model
        from espnet.nets.asr_interface import ASRInterface
        model, train_args = load_trained_model(self.asr_prob_model)
        assert isinstance(model, ASRInterface)
        model.recog_args = self.args
        # xs_pad = (32, 789, 83)
        # make 64 fbank -> 1 fbank for decoder per one fbank
        # like batch
        xs_pad_batch_len = xs_pad.shape[0]
        feat_max_len = xs_pad.shape[1]
        n_mel = xs_pad.shape[2]
        # doing decoding with asr
        # [732] * 32
        lens = [feat_max_len] * xs_pad_batch_len
        hyps_list = []
        sentences_scores_list = []
        for i in range(xs_pad_batch_len):
            # 0-31
            feat = []
            for j in range(feat_max_len):
                # j : 0- 732
                xp = xs_pad[i][j] # len(xp) = 83
                su = torch.sum(xp)
                xp_l = xp.tolist()
                if su == 0:
                    if xp_l == [0] * n_mel:
                        break
                    else:
                        feat.append(xp_l)
                else:
                    feat.append(xp_l)
            # list to tensor
            feat = torch.tensor(feat) # 732, 83
            # one feat -> asr model
            nbest_hyps, local_scores_list = model.recognize(
                feat, self.args, self.args.char_list, self.args.rnnlm
            )
            hyps_list.append(nbest_hyps) # exclude first sos 7806
            sentences_scores_list.append(local_scores_list)
        from copy import deepcopy
        hyps_list_pre = deepcopy(hyps_list)
        sentences_scores_list_pre = deepcopy(sentences_scores_list)
        '''
        # [ADD] use self.soft_label
        # hyps_list = []
        # sentences_scores_list = []
        # for uttid in uttid_list:
            # [WILLFIX]
            # path_train_pickle = path_train / uttid
            # path_dev_pickle = path_dev / uttid
            # nbest_hyps, _local_scores_list = self.soft_label[uttid]
                # with path_dev_pickle.open(mode='rb') as f:
                #     nbest_hyps, _local_scores_list = pickle.load(f)
                #  [WILLFIX] sorry inconvenient...
        #    local_scores_list = _local_scores_list[0]
        #    hyps_list.append(nbest_hyps)
        #    sentences_scores_list.append(local_scores_list)
        # [ADD] load hyps_list, local_scores_list
        # pre-decoding load version
        # load_from_dir = True
        # if load_from_dir == True:
        hyps_list = []
        sentences_scores_list = []
        pre_decoding_dir = self.args.pre_decoding_dir
        path1 = Path()
        path2 = Path()
        path_train = path1 / pre_decoding_dir / 'train'
        path_dev = path2 / pre_decoding_dir / 'dev'

        for uttid in uttid_list:
            # [WILLFIX]
            path_train_pickle = path_train / uttid
            path_dev_pickle = path_dev / uttid
            if path_train_pickle.exists():
                with path_train_pickle.open(mode='rb') as f:
                    nbest_hyps, _local_scores_list = pickle.load(f)
            elif path_dev_pickle.exists():
                with path_dev_pickle.open(mode='rb') as f:
                    nbest_hyps, _local_scores_list = pickle.load(f)
                # [WILLFIX] sorry inconvenient...
            local_scores_list = _local_scores_list[0]

            hyps_list.append(nbest_hyps)
            sentences_scores_list.append(local_scores_list)

        # for test
        # assert hyps_list == hyps_list_pre
        # assert sentences_scores_list == sentences_scores_list_pre

        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        # 2. forward decoder
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        # replace <sos> with target language ID
        if self.replace_sos:
            ys_in_pad = torch.cat([tgt_lang_ids, ys_in_pad[:, 1:]], dim=1)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)
        pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

        # 3. compute ST loss
        loss_att = self.criterion_st(pred_pad, ys_out_pad)

        self.acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )

        # 4. compute corpus-level bleu in a mini-batch
        if self.training:
            self.bleu = None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # 5. compute auxiliary ASR loss
        # [ADD]
        self.hyps_list = hyps_list
        self.sentences_scores_list = sentences_scores_list
        soft_tgt_weight = self.args.soft_tgt_weight

        loss_asr_att, acc_asr, loss_asr_ctc, cer_ctc, cer, wer = self.forward_asr(
            hs_pad, hs_mask, ys_pad_src, hyps_list, sentences_scores_list, soft_tgt_weight 
        )

        # 6. compute auxiliary MT loss
        loss_mt, acc_mt = 0.0, None
        if self.mt_weight > 0:
            loss_mt, acc_mt = self.forward_mt(
                ys_pad_src, ys_in_pad, ys_out_pad, ys_mask
            )

        asr_ctc_weight = self.mtlalpha
        self.loss = (
            (1 - self.asr_weight - self.mt_weight) * loss_att
            + self.asr_weight
            * (asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att)
            + self.mt_weight * loss_mt
        )
        loss_asr_data = float(
            asr_ctc_weight * loss_asr_ctc + (1 - asr_ctc_weight) * loss_asr_att
        )
        loss_mt_data = None if self.mt_weight == 0 else float(loss_mt)
        loss_st_data = float(loss_att)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_asr_data,
                loss_mt_data,
                loss_st_data,
                acc_asr,
                acc_mt,
                self.acc,
                cer_ctc,
                cer,
                wer,
                self.bleu,
                loss_data,
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    # [ADD]
    # def forward_asr(self, hs_pad, hs_mask, ys_pad):
    def forward_asr(self, hs_pad, hs_mask, ys_pad, hyps_list, sentences_scores_list, soft_tgt_weight): # size : batch size
        # change here
        """Forward pass in the auxiliary ASR task.

        :param torch.Tensor hs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_mask: batch of input token mask (B, Lmax)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ASR attention loss value
        :rtype: torch.Tensor
        :return: accuracy in ASR attention decoder
        :rtype: float
        :return: ASR CTC loss value
        :rtype: torch.Tensor
        :return: character error rate from CTC prediction
        :rtype: float
        :return: character error rate from attetion decoder prediction
        :rtype: float
        :return: word error rate from attetion decoder prediction
        :rtype: float
        """
        loss_att, loss_ctc = 0.0, 0.0
        acc = None
        cer, wer = None, None
        cer_ctc = None
        if self.asr_weight == 0:
            return loss_att, acc, loss_ctc, cer_ctc, cer, wer
        # attention
        if self.mtlalpha < 1:
            # ys is hidden states
            ys_in_pad_asr, ys_out_pad_asr = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask_asr = target_mask(ys_in_pad_asr, self.ignore_id)
            pred_pad, _ = self.decoder_asr(ys_in_pad_asr, ys_mask_asr, hs_pad, hs_mask)
            # [ADD] soft - hart tgt loss
            loss_att = self.criterion_asr(pred_pad, ys_out_pad_asr, hyps_list, sentences_scores_list, soft_tgt_weight)
            acc = th_accuracy(
                pred_pad.view(-1, self.odim),
                ys_out_pad_asr,
                ignore_label=self.ignore_id,
            )
            if not self.training:
                ys_hat_asr = pred_pad.argmax(dim=-1)
                cer, wer = self.error_calculator_asr(ys_hat_asr.cpu(), ys_pad.cpu())
        # CTC
        if self.mtlalpha > 0:
            batch_size = hs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if not self.training:
                ys_hat_ctc = self.ctc.argmax(
                    hs_pad.view(batch_size, -1, self.adim)
                ).data
                cer_ctc = self.error_calculator_asr(
                    ys_hat_ctc.cpu(), ys_pad.cpu(), is_ctc=True
                )
                # for visualization
                self.ctc.softmax(hs_pad)
        return loss_att, acc, loss_ctc, cer_ctc, cer, wer

    def forward_mt(self, xs_pad, ys_in_pad, ys_out_pad, ys_mask):
        """Forward pass in the auxiliary MT task.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ys_in_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_out_pad: batch of padded target sequences (B, Lmax)
        :param torch.Tensor ys_mask: batch of input token mask (B, Lmax)
        :return: MT loss value
        :rtype: torch.Tensor
        :return: accuracy in MT decoder
        :rtype: float
        """
        loss, acc = 0.0, None
        if self.mt_weight == 0:
            return loss, acc

        ilens = torch.sum(xs_pad != self.ignore_id, dim=1).cpu().numpy()
        # NOTE: xs_pad is padded with -1
        xs = [x[x != self.ignore_id] for x in xs_pad]  # parse padded xs
        xs_zero_pad = pad_list(xs, self.pad)  # re-pad with zero
        xs_zero_pad = xs_zero_pad[:, : max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_zero_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder_mt(xs_zero_pad, src_mask)
        pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        loss = self.criterion_mt(pred_pad, ys_out_pad)
        acc = th_accuracy(
            pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
        )
        return loss, acc

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder)

    def encode(self, x):
        """Encode source acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def translate(
        self,
        x,
        trans_args,
        char_list=None,
        rnnlm=None,
        use_jit=False,
    ):
        """Translate input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace trans_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # preprate sos
        if getattr(trans_args, "tgt_lang", False):
            if self.replace_sos:
                y = char_list.index(trans_args.tgt_lang)
        else:
            y = self.sos
        logging.info("<sos> index: " + str(y))
        logging.info("<sos> mark: " + char_list[y])

        enc_output = self.encode(x).unsqueeze(0)
        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = trans_args.beam_size
        penalty = trans_args.penalty

        vy = h.new_zeros(1).long()

        if trans_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(trans_args.maxlenratio * h.size(0)))
        minlen = int(trans_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + trans_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam, dim=1
                )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += trans_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and trans_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), trans_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform translation "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            trans_args = Namespace(**vars(trans_args))
            trans_args.minlenratio = max(0.0, trans_args.minlenratio - 0.1)
            return self.translate(x, trans_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    # [ADD] uttid_list
    # def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src, uttid_list):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            # [ADD] uttid_list
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src, uttid_list)
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention) and m.attn is not None
            ):  # skip MHA for submodules
                ret[name] = m.attn.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self, xs_pad, ilens, ys_pad, ys_pad_src):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :param torch.Tensor ys_pad_src:
            batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.asr_weight == 0 or self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
        ret = None
        for name, m in self.named_modules():
            if isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
