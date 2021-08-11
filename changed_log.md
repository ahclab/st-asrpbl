## changed log

The information would find with `git diff --find-copies-harder d8e1bf011c4d791a5898b72f0f3f15eab3c95450`

## changed files 
- <CHANGED_FILE>
    - <COPIED_FROM>

- egs/fisher_callhome_spanish/asr1b/conf/decode-check-st-asrpbl.yaml
	- egs/fisher_callhome_spanish/asr1b/conf/decode.yaml
- egs/fisher_callhome_spanish/asr1b/conf/decode-st-asrpbl.yaml
	- egs/fisher_callhome_spanish/asr1b/conf/decode.yaml
- egs/fisher_callhome_spanish/asr1b/conf/train-pretrain-asr-for-pbl.yaml
    - egs/fisher_callhome_spanish/asr1b/conf/tuning/train_pytorch_transformer_bpe.yaml
- egs/fisher_callhome_spanish/asr1b/run-proposed.sh
	- egs/fisher_callhome_spanish/asr1b/run.sh
        - Training with ST dict
```
-dict=data/lang_1spm/${train_set}_${bpemode}${nbpe}_units_${case}.txt
-nlsyms=data/lang_1spm/${train_set}_non_lang_syms_${case}.txt
-bpemodel=data/lang_1spm/${train_set}_${bpemode}${nbpe}_${case}
+# use st dictionary
+dict=../st1/data/lang_1spm/train_sp.en_bpe1000_units_lc.rm.txt
+nlsyms=../st1/data/lang_1spm/train_sp.en_non_lang_syms_lc.rm.txt
+bpemodel=../st1/data/lang_1spm/train_sp.en_bpe1000_lc.rm
```
- egs/fisher_callhome_spanish/st1/conf/decode-st-asrpbl.yaml
	- egs/fisher_callhome_spanish/st1/conf/decode.yaml
- egs/fisher_callhome_spanish/st1/conf/pre_decoding.yaml
	- egs/fisher_callhome_spanish/asr1b/conf/decode.yaml
- egs/fisher_callhome_spanish/st1/conf/train-st-asrpbl.yaml
	- egs/libri_trans/st1/conf/tuning/train_pytorch_transformer_bpe.yaml
- egs/fisher_callhome_spanish/st1/run-proposed.sh
	- egs/fisher_callhome_spanish/st1/run.sh
		- parameter for asrpbl
```
+# [ADD] configuration
+data=data
+pre_decoding_recog_model=../asr1b/exp/pretrain-asr-for-pbl/results/model.val1.avg.best
+pre_decoding_dir=../asr1b/exp/pretrain-asr-for-pbl/pre_decoding
+model_module="espnet.nets.pytorch_backend.e2e_st_transformer_asrpbl:E2E"
+epochs=30
+asr_weight=0.4
+lsm_weight_st=0.1
+lsm_weight_asr=0.1
+soft_tgt_weight=0.5
+
```
- Preparing Pre-decoding soft labels

```
+asr_nj=4
+
+if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
+    echo "stage 3: Pre-Decoding dev train"
+    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
+       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
+
+        # [ADD] set ASR model for asrpbl
+        pre_decoding_recog_model=${pre_decoding_recog_model}
+    fi
+    pids=() # initialize pids
+    
+    # for mode in dev; do # test
+    for mode in dev train; do
+    (
+        if [ $mode = train ]; then
+            feat_recog_dir=${dumpdir}/train_sp.en/deltafalse
+        elif [ $mode = dev ]; then
+            feat_recog_dir=${dumpdir}/train_dev.en/deltafalse
+        fi
+        pre_decoding_mode_dir=${pre_decoding_dir}/${mode}
+        asr_decode_config=conf/pre_decoding.yaml
+
+        # split data
+        if [ ${mode} = train ]; then
+            splitjson.py --parts ${asr_nj} ${dumpdir}/train_sp.en/deltafalse/data_bpe1000.lc.rm_lc.rm.json
+        elif [ ${mode} = dev ]; then
+            splitjson.py --parts ${asr_nj} ${dumpdir}/train_dev.en/deltafalse/data_bpe1000.lc.rm_lc.rm.json
+        fi
+
+
+        #### use CPU for decoding
+        ngpu=0
+
+        # [FOR DEBUG]
+        ${decode_cmd} JOB=1:${asr_nj} ${expdir}/${pre_decoding_mode_dir}/log/decode.JOB.log \
+            asr_recog_pre_decoding.py \
+            --config ${asr_decode_config} \
+            --ngpu ${ngpu} \
+            --backend ${backend} \
+            --batchsize 0 \
+            --recog-json ${feat_recog_dir}/split${asr_nj}utt/data_${bpemode}${nbpe}.JOB.json \
+            --result-label ${pre_decoding_mode_dir}/data.JOB.json \
+            --model ${pre_decoding_recog_model} \
+            --decode-dir ${pre_decoding_mode_dir}
+    ) &
+    pids+=($!) # store background pids
+    done
+    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
+    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
+    echo "Finished"
+fi
```
- Training ST with soft labels
```

 if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
     echo "stage 4: Network Training"
 
-    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
-        st_train.py \
+    # python espnet-link/bin/st_train_pbl.py
+    ${cuda_cmd} --gpu ${ngpu} ${expdir}/${timestamp}_stage${stage}_train.log \
+        st_train_pbl.py \
         --config ${train_config} \
         --preprocess-conf ${preprocess_config} \
         --ngpu ${ngpu} \
@@ -297,7 +366,14 @@ if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
         --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
         --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.${src_case}_${tgt_case}.json \
         --enc-init ${asr_model} \
-        --dec-init ${mt_model}
+        --dec-init ${mt_model} \
+        --epochs ${epochs} \
+        --model-module ${model_module} \
+        --pre-decoding-dir ${pre_decoding_dir} \
+        --asr-weight ${asr_weight} \
+        --lsm-weight-st ${lsm_weight_st} \
+        --lsm-weight-asr ${lsm_weight_asr} \
+        --soft-tgt-weight ${soft_tgt_weight}
 fi
```
- egs/fisher_callhome_spanish/st1/run-st-asrpbl.sh
	- egs/fisher_callhome_spanish/st1/run.sh
- espnet/asr/asr_utils_pre_decoding.py
	- espnet/asr/asr_utils.py
```

--- a/espnet/asr/asr_utils.py
+++ b/espnet/asr/asr_utils_pre_decoding.py
@@ -829,7 +829,9 @@ def add_results_to_json(js, nbest_hyps, char_list):
 
         # copy ground-truth
         if len(js["output"]) > 0:
-            out_dic = dict(js["output"][0].items())
+            # [FIX] for train and dev set (id0 -> target1,en / id1 -> target2,es)
+            # [BEFORE] out_dic = dict(js["output"][0].items()) # for test data (id0 -> es)
+            out_dic = dict(js["output"][1].items())
         else:
             # for no reference case (e.g., speech translation)
             out_dic = {"name": ""}
```

- espnet/asr/pytorch_backend/asr_init_pre_decoding.py
	- espnet/asr/pytorch_backend/asr_init.py
```
from espnet.asr.asr_utils import adadelta_eps_decay
-from espnet.asr.asr_utils import add_results_to_json
+# [ADD]
+# [BEFORE] from espnet.asr.asr_utils import add_results_to_json
+from espnet.asr.asr_utils_pre_decoding import add_results_to_json
 from espnet.asr.asr_utils import CompareValueTrigger
 from espnet.asr.asr_utils import format_mulenc_args
 from espnet.asr.asr_utils import get_model_conf
```
```
@@ -1030,9 +1032,25 @@ def recog(args):
                         feat, args, train_args.char_list
                     )
                 else:
-                    nbest_hyps = model.recognize(
+                    # [ADD] local_scores_list(per one sentence)
+                    # [BEFORE] nbest_hyps = model.recognize(
+                    nbest_hyps, local_scores_list = model.recognize_pre_decoding(
                         feat, args, train_args.char_list, rnnlm
                     )
+                # [ADD] pickle save in decoder_dir
+                from pathlib import Path
+                import pickle
+                p = Path()
+                output_dir = p / args.decode_dir
+                if not output_dir.exists():
+                    output_dir.mkdir(parents=True)
+
+                decoder_output = [nbest_hyps, [local_scores_list]]
+                pickle_path = output_dir / name
+                if not pickle_path.exists():
+                    with pickle_path.open(mode='wb') as f:
+                        pickle.dump(decoder_output, f)
+
                 new_js[name] = add_results_to_json(
                     js[name], nbest_hyps, train_args.char_list
                 )

```
- espnet/asr/pytorch_backend/asr_pre_decoding.py
	- espnet/asr/pytorch_backend/asr.py
```
@@ -23,7 +23,9 @@ import torch
 from torch.nn.parallel import data_parallel
 
 from espnet.asr.asr_utils import adadelta_eps_decay
-from espnet.asr.asr_utils import add_results_to_json
+# [ADD]
+# [BEFORE] from espnet.asr.asr_utils import add_results_to_json
+from espnet.asr.asr_utils_pre_decoding import add_results_to_json
 from espnet.asr.asr_utils import CompareValueTrigger
 from espnet.asr.asr_utils import format_mulenc_args
 from espnet.asr.asr_utils import get_model_conf
@@ -1030,9 +1032,25 @@ def recog(args):
                         feat, args, train_args.char_list
                     )
                 else:
-                    nbest_hyps = model.recognize(
+                    # [ADD] local_scores_list(per one sentence)
+                    # [BEFORE] nbest_hyps = model.recognize(
+                    nbest_hyps, local_scores_list = model.recognize_pre_decoding(
                         feat, args, train_args.char_list, rnnlm
                     )
+                # [ADD] pickle save in decoder_dir
+                from pathlib import Path
+                import pickle
+                p = Path()
+                output_dir = p / args.decode_dir
+                if not output_dir.exists():
+                    output_dir.mkdir(parents=True)
+
+                decoder_output = [nbest_hyps, [local_scores_list]]
+                pickle_path = output_dir / name
+                if not pickle_path.exists():
+                    with pickle_path.open(mode='wb') as f:
+                        pickle.dump(decoder_output, f)
+
                 new_js[name] = add_results_to_json(
                     js[name], nbest_hyps, train_args.char_list
                 )
```
- espnet/bin/asr_recog_pre_decoding.py
	- espnet/bin/asr_recog.py
```

+    #[ADD]
+    parser.add_argument(
+        "--decode-dir",
+        type=str,
+        default=None,
+        help="decoder_dir",
+    )
+
 
```

```
@@ -296,8 +304,9 @@ def main(args):
 
                     recog_v2(args)
                 else:
-                    from espnet.asr.pytorch_backend.asr import recog
-
+                    # from espnet.asr.pytorch_backend.asr import recog
+                    # [ADD]
+                    from espnet.asr.pytorch_backend.asr_pre_decoding import recog
                     if args.dtype != "float32":
                         raise NotImplementedError(
                             f"`--dtype {args.dtype}` is only available with `--api v2`"

```
- espnet/bin/st_train_pbl.py
	- espnet/bin/st_train.py

```
+    # [ADD] distinguish lsm-weight for ST-task and ASR-task
+    # [BEFORE]
+    # parser.add_argument(
+    #     "--lsm-weight", default=0.0, type=float, help="Label smoothing weight"
+    # )
     parser.add_argument(
-        "--lsm-weight", default=0.0, type=float, help="Label smoothing weight"
+        "--lsm-weight-st", default="", required=required, type=float, help="Label smoothing weight for ST-task"
+    )
+    parser.add_argument(
+        "--lsm-weight-asr", default="", required=required, type=float, help="Label smoothing weight for ASR-task"
+    )
+    parser.add_argument(
+        "--lsm-weight-mt", default="", required=required, type=float, help="Label smoothing weight for MT-task"
     )
     # recognition options to compute CER/WER
     parser.add_argument(
@@ -438,6 +449,13 @@ def get_parser(parser=None, required=True):
     )
     parser.add_argument("--fbank-fmin", type=float, default=0.0, help="")
     parser.add_argument("--fbank-fmax", type=float, default=None, help="")
+    # [ADD]
+    parser.add_argument(
+        "--pre-decoding-dir", default=None, required=required, type=str, help="dirpath includes hypothesis and local_scores_list by pre_decoding",
+    )
+    parser.add_argument(
+        "--soft-tgt-weight", default=None, required=required, type=float, help="soft-tgt-weight for asrpbl (0.0-1.0)"
+    )
     return parser
```
```
 
     if args.backend == "pytorch":
-        from espnet.st.pytorch_backend.st import train
+        # [ADD]
+        # from espnet.st.pytorch_backend.st import train
+        from espnet.st.pytorch_backend.st_pbl import train
 
         train(args)
```
- espnet/nets/pytorch_backend/e2e_asr_transformer.py
	- espnet/nets/pytorch_backend/e2e_asr_transformer.py
```
+
+    # [ADD] based on recognize
+    def recognize_pre_decoding(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
+        """Recognize input speech.
+
+        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
+        :param Namespace recog_args: argment Namespace contraining options
+        :param list char_list: list of characters
+        :param torch.nn.Module rnnlm: language model module
+        :return: N-best decoding results
+        :rtype: list
+        """
+        enc_output = self.encode(x).unsqueeze(0)
+        if self.mtlalpha == 1.0:
+            recog_args.ctc_weight = 1.0
+            logging.info("Set to pure CTC decoding mode.")
+
+        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
+            from itertools import groupby
+
+            lpz = self.ctc.argmax(enc_output)
+            collapsed_indices = [x[0] for x in groupby(lpz[0])]
+            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
+            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
+            if recog_args.beam_size > 1:
+                raise NotImplementedError("Pure CTC beam search is not implemented.")
+            # TODO(hirofumi0810): Implement beam search
+            return nbest_hyps
+        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
+            lpz = self.ctc.log_softmax(enc_output)
+            lpz = lpz.squeeze(0)
+        else:
+            lpz = None
+
+        h = enc_output.squeeze(0)
+
+        logging.info("input lengths: " + str(h.size(0)))
+        # search parms
+        beam = recog_args.beam_size
+        penalty = recog_args.penalty
+        ctc_weight = recog_args.ctc_weight
+
+        # preprare sos
+        y = self.sos
+        vy = h.new_zeros(1).long()
+
+        if recog_args.maxlenratio == 0:
+            maxlen = h.shape[0]
+        else:
+            # maxlen >= 1
+            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
+        minlen = int(recog_args.minlenratio * h.size(0))
+        logging.info("max output length: " + str(maxlen))
+        logging.info("min output length: " + str(minlen))
+
+        # initialize hypothesis
+        if rnnlm:
+            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
+        else:
+            hyp = {"score": 0.0, "yseq": [y]}
+        if lpz is not None:
+            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
+            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
+            hyp["ctc_score_prev"] = 0.0
+            if ctc_weight != 1.0:
+                # pre-pruning based on attention scores
+                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
+            else:
+                ctc_beam = lpz.shape[-1]
+        hyps = [hyp]
+        ended_hyps = []
+
+        import six
+
+        traced_decoder = None
+        # [ADD]
+        # posterior list of one sentence
+        local_scores_list = []
+
+        for i in six.moves.range(maxlen):
+            logging.debug("position " + str(i))
+
+            hyps_best_kept = []
+            for hyp in hyps:
+                vy[0] = hyp["yseq"][i]
+
+                # get nbest local scores and their ids
+                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
+                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
+                # FIXME: jit does not match non-jit result
+                if use_jit:
+                    if traced_decoder is None:
+                        traced_decoder = torch.jit.trace(
+                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
+                        )
+                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
+                else:
+                    local_att_scores = self.decoder.forward_one_step(
+                        ys, ys_mask, enc_output
+                    )[0]
+
+                if rnnlm:
+                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
+                    local_scores = (
+                        local_att_scores + recog_args.lm_weight * local_lm_scores
+                    )
+                else:
+                    local_scores = local_att_scores
+
+                if lpz is not None:
+                    local_best_scores, local_best_ids = torch.topk(
+                        local_att_scores, ctc_beam, dim=1
+                    )
+                    ctc_scores, ctc_states = ctc_prefix_score(
+                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
+                    )
+                    local_scores = (1.0 - ctc_weight) * local_att_scores[
+                        :, local_best_ids[0]
+                    ] + ctc_weight * torch.from_numpy(
+                        ctc_scores - hyp["ctc_score_prev"]
+                    )
+                    if rnnlm:
+                        local_scores += (
+                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
+                        )
+                    local_best_scores, joint_best_ids = torch.topk(
+                        local_scores, beam, dim=1
+                    )
+                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
+                else:
+                    local_best_scores, local_best_ids = torch.topk(
+                        local_scores, beam, dim=1
+                    )
+                    # [ADD]
+                    assert len(local_scores) == 1
+                    #local_scores_list.append(local_scores[0])
+                    local_scores_list.append(local_scores[0].tolist())
+
+                for j in six.moves.range(beam):
+                    new_hyp = {}
+                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
+                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
+                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
+                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
+                    if rnnlm:
+                        new_hyp["rnnlm_prev"] = rnnlm_state
+                    if lpz is not None:
+                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
+                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
+                    # will be (2 x beam) hyps at most
+                    hyps_best_kept.append(new_hyp)
+
+                hyps_best_kept = sorted(
+                    hyps_best_kept, key=lambda x: x["score"], reverse=True
+                )[:beam]
+
+            # sort and get nbest
+            hyps = hyps_best_kept
+            logging.debug("number of pruned hypothes: " + str(len(hyps)))
+            if char_list is not None:
+                logging.debug(
+                    "best hypo: "
+                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
+                )
+
+            # add eos in the final loop to avoid that there are no ended hyps
+            if i == maxlen - 1:
+                logging.info("adding <eos> in the last postion in the loop")
+                for hyp in hyps:
+                    hyp["yseq"].append(self.eos)
+
+            # add ended hypothes to a final list, and removed them from current hypothes
+            # (this will be a probmlem, number of hyps < beam)
+            remained_hyps = []
+            for hyp in hyps:
+                if hyp["yseq"][-1] == self.eos:
+                    # only store the sequence that has more than minlen outputs
+                    # also add penalty
+                    if len(hyp["yseq"]) > minlen:
+                        hyp["score"] += (i + 1) * penalty
+                        if rnnlm:  # Word LM needs to add final <eos> score
+                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
+                                hyp["rnnlm_prev"]
+                            )
+                        ended_hyps.append(hyp)
+                else:
+                    remained_hyps.append(hyp)
+
+            # end detection
+            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
+                logging.info("end detected at %d", i)
+                break
+
+            hyps = remained_hyps
+            if len(hyps) > 0:
+                logging.debug("remeined hypothes: " + str(len(hyps)))
+            else:
+                logging.info("no hypothesis. Finish decoding.")
+                break
+
+            if char_list is not None:
+                for hyp in hyps:
+                    logging.debug(
+                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
+                    )
+
+            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))
+
+        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
+            : min(len(ended_hyps), recog_args.nbest)
+        ]
+
+        # check number of hypotheis
+        if len(nbest_hyps) == 0:
+            logging.warning(
+                "there is no N-best results, perform recognition "
+                "again with smaller minlenratio."
+            )
+            # should copy becasuse Namespace will be overwritten globally
+            recog_args = Namespace(**vars(recog_args))
+            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
+            return self.recognize(x, recog_args, char_list, rnnlm)
+
+        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
+        logging.info(
+            "normalized log probability: "
+            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
+        )
+        # [ADD] local_scores_list (per one sentence)
+        # [BEFORE] return nbest_hyps
+        return nbest_hyps, local_scores_list
```

- espnet/nets/pytorch_backend/e2e_asr_transformer_pre_decoding.py
	- espnet/nets/pytorch_backend/e2e_asr_transformer.py
```
@@ -495,6 +495,10 @@ class E2E(ASRInterface, torch.nn.Module):
         import six
 
         traced_decoder = None
+        # [ADD]
+        # posterior list of one sentence
+        local_scores_list = []
+
         for i in six.moves.range(maxlen):
             logging.debug("position " + str(i))
 
@@ -549,6 +553,10 @@ class E2E(ASRInterface, torch.nn.Module):
                     local_best_scores, local_best_ids = torch.topk(
                         local_scores, beam, dim=1
                     )
+                    # [ADD]
+                    assert len(local_scores) == 1
+                    #local_scores_list.append(local_scores[0])
+                    local_scores_list.append(local_scores[0].tolist())
 
                 for j in six.moves.range(beam):
                     new_hyp = {}
@@ -640,7 +648,9 @@ class E2E(ASRInterface, torch.nn.Module):
             "normalized log probability: "
             + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
         )
-        return nbest_hyps
+        # [ADD] local_scores_list (per one sentence)
+        # [BEFORE] return nbest_hyps
+        return nbest_hyps, local_scores_list

```
- espnet/nets/pytorch_backend/e2e_st_transformer_asrpbl.py
	- espnet/nets/pytorch_backend/e2e_st_transformer.py
```
@@ -11,6 +11,11 @@ from distutils.util import strtobool
 import logging
 import math
 
+# [ADD]
+from collections import Counter
+from pathlib import Path
+import pickle
+
 import torch
 
 from espnet.nets.e2e_asr_common import end_detect
@@ -31,6 +36,10 @@ from espnet.nets.pytorch_backend.transformer.initializer import initialize
 from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
     LabelSmoothingLoss,  # noqa: H301
 )
+# [ADD]
+from espnet.nets.pytorch_backend.transformer.posterior_based_loss import (
+    PosteriorBasedLoss,  # noqa: H301
+)
 from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
 from espnet.nets.pytorch_backend.transformer.mask import target_mask
 from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
@@ -255,11 +264,25 @@ class E2E(STInterface, torch.nn.Module):
         self.ignore_id = ignore_id
         self.subsample = get_subsample(args, mode="st", arch="transformer")
         self.reporter = Reporter()
-
-        self.criterion = LabelSmoothingLoss(
+        # [ADD]
+        self.batch_size = args.batch_size
+        self.criterion_st = LabelSmoothingLoss(
+            self.odim,
+            self.ignore_id,
+            args.lsm_weight_st,
+            args.transformer_length_normalized_loss,
+        )
+        # asrpbl
+        self.criterion_asr = PosteriorBasedLoss(
+            self.odim,
+            self.ignore_id,
+            args.lsm_weight_asr,
+            args.transformer_length_normalized_loss,
+        )
+        self.criterion_mt = LabelSmoothingLoss(
             self.odim,
             self.ignore_id,
-            args.lsm_weight,
+            args.lsm_weight_mt,
             args.transformer_length_normalized_loss,
         )
         # submodule for ASR task
@@ -306,7 +329,6 @@ class E2E(STInterface, torch.nn.Module):
         self.error_calculator = MTErrorCalculator(
             args.char_list, args.sym_space, args.sym_blank, args.report_bleu
         )
-
         # recognition error calculator
         self.error_calculator_asr = ASRErrorCalculator(
             args.char_list,
@@ -320,6 +342,8 @@ class E2E(STInterface, torch.nn.Module):
         # multilingual E2E-ST related
         self.multilingual = getattr(args, "multilingual", False)
         self.replace_sos = getattr(args, "replace_sos", False)
+        # [ADD]
+        self.args = args
 
     def reset_parameters(self, args):
         """Initialize parameters."""
@@ -330,7 +354,9 @@ class E2E(STInterface, torch.nn.Module):
             )
             torch.nn.init.constant_(self.encoder_mt.embed[0].weight[self.pad], 0)
 
-    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src):
+    # [WILL ADD -> ADD] batch more information : uttid
+    # [ADD] uttid_list
+    def forward(self, xs_pad, ilens, ys_pad, ys_pad_src, uttid_list):
         """E2E forward.
 
         :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
@@ -351,7 +377,101 @@ class E2E(STInterface, torch.nn.Module):
             ys_pad = ys_pad[:, 1:]  # remove target language ID in the beggining
 
         # 1. forward encoder
+        # this is mel batch
         xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
+        self.xs_pad = xs_pad
+        '''
+        # not pre-decoding version
+        from torch.nn.utils.rnn import pack_padded_sequence
+        from espnet.asr.pytorch_backend.asr_init import load_trained_model
+        from espnet.nets.asr_interface import ASRInterface
+        model, train_args = load_trained_model(self.asr_prob_model)
+        assert isinstance(model, ASRInterface)
+        model.recog_args = self.args
+        # xs_pad = (32, 789, 83)
+        # make 64 fbank -> 1 fbank for decoder per one fbank
+        # like batch
+        xs_pad_batch_len = xs_pad.shape[0]
+        feat_max_len = xs_pad.shape[1]
+        n_mel = xs_pad.shape[2]
+        # doing decoding with asr
+        # [732] * 32
+        lens = [feat_max_len] * xs_pad_batch_len
+        hyps_list = []
+        sentences_scores_list = []
+        for i in range(xs_pad_batch_len):
+            # 0-31
+            feat = []
+            for j in range(feat_max_len):
+                # j : 0- 732
+                xp = xs_pad[i][j] # len(xp) = 83
+                su = torch.sum(xp)
+                xp_l = xp.tolist()
+                if su == 0:
+                    if xp_l == [0] * n_mel:
+                        break
+                    else:
+                        feat.append(xp_l)
+                else:
+                    feat.append(xp_l)
+            # list to tensor
+            feat = torch.tensor(feat) # 732, 83
+            # one feat -> asr model
+            nbest_hyps, local_scores_list = model.recognize(
+                feat, self.args, self.args.char_list, self.args.rnnlm
+            )
+            hyps_list.append(nbest_hyps) # exclude first sos 7806
+            sentences_scores_list.append(local_scores_list)
+        from copy import deepcopy
+        hyps_list_pre = deepcopy(hyps_list)
+        sentences_scores_list_pre = deepcopy(sentences_scores_list)
+        '''
+        # [ADD] use self.soft_label
+        # hyps_list = []
+        # sentences_scores_list = []
+        # for uttid in uttid_list:
+            # [WILLFIX]
+            # path_train_pickle = path_train / uttid
+            # path_dev_pickle = path_dev / uttid
+            # nbest_hyps, _local_scores_list = self.soft_label[uttid]
+                # with path_dev_pickle.open(mode='rb') as f:
+                #     nbest_hyps, _local_scores_list = pickle.load(f)
+                #  [WILLFIX] sorry inconvenient...
+        #    local_scores_list = _local_scores_list[0]
+        #    hyps_list.append(nbest_hyps)
+        #    sentences_scores_list.append(local_scores_list)
+        # [ADD] load hyps_list, local_scores_list
+        # pre-decoding load version
+        # load_from_dir = True
+        # if load_from_dir == True:
+        hyps_list = []
+        sentences_scores_list = []
+        pre_decoding_dir = self.args.pre_decoding_dir
+        path1 = Path()
+        path2 = Path()
+        path_train = path1 / pre_decoding_dir / 'train'
+        path_dev = path2 / pre_decoding_dir / 'dev'
+
+        for uttid in uttid_list:
+            # [WILLFIX]
+            path_train_pickle = path_train / uttid
+            path_dev_pickle = path_dev / uttid
+            if path_train_pickle.exists():
+                with path_train_pickle.open(mode='rb') as f:
+                    nbest_hyps, _local_scores_list = pickle.load(f)
+            elif path_dev_pickle.exists():
+                with path_dev_pickle.open(mode='rb') as f:
+                    nbest_hyps, _local_scores_list = pickle.load(f)
+                # [WILLFIX] sorry inconvenient...
+            local_scores_list = _local_scores_list[0]
+
+            hyps_list.append(nbest_hyps)
+            sentences_scores_list.append(local_scores_list)
+
+        # for test
+        # assert hyps_list == hyps_list_pre
+        # assert sentences_scores_list == sentences_scores_list_pre
+
         src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
         hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
 
@@ -364,7 +484,7 @@ class E2E(STInterface, torch.nn.Module):
         pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
 
         # 3. compute ST loss
-        loss_att = self.criterion(pred_pad, ys_out_pad)
+        loss_att = self.criterion_st(pred_pad, ys_out_pad)
 
         self.acc = th_accuracy(
             pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
@@ -378,8 +498,13 @@ class E2E(STInterface, torch.nn.Module):
             self.bleu = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
 
         # 5. compute auxiliary ASR loss
+        # [ADD]
+        self.hyps_list = hyps_list
+        self.sentences_scores_list = sentences_scores_list
+        soft_tgt_weight = self.args.soft_tgt_weight
+
         loss_asr_att, acc_asr, loss_asr_ctc, cer_ctc, cer, wer = self.forward_asr(
-            hs_pad, hs_mask, ys_pad_src
+            hs_pad, hs_mask, ys_pad_src, hyps_list, sentences_scores_list, soft_tgt_weight 
         )
 
         # 6. compute auxiliary MT loss
@@ -421,7 +546,10 @@ class E2E(STInterface, torch.nn.Module):
             logging.warning("loss (=%f) is not correct", loss_data)
         return self.loss
 
-    def forward_asr(self, hs_pad, hs_mask, ys_pad):
+    # [ADD]
+    # def forward_asr(self, hs_pad, hs_mask, ys_pad):
+    def forward_asr(self, hs_pad, hs_mask, ys_pad, hyps_list, sentences_scores_list, soft_tgt_weight): # size : batch size
+        # change here
         """Forward pass in the auxiliary ASR task.
 
         :param torch.Tensor hs_pad: batch of padded source sequences (B, Tmax, idim)
@@ -446,16 +574,16 @@ class E2E(STInterface, torch.nn.Module):
         cer_ctc = None
         if self.asr_weight == 0:
             return loss_att, acc, loss_ctc, cer_ctc, cer, wer
-
         # attention
         if self.mtlalpha < 1:
+            # ys is hidden states
             ys_in_pad_asr, ys_out_pad_asr = add_sos_eos(
                 ys_pad, self.sos, self.eos, self.ignore_id
             )
             ys_mask_asr = target_mask(ys_in_pad_asr, self.ignore_id)
             pred_pad, _ = self.decoder_asr(ys_in_pad_asr, ys_mask_asr, hs_pad, hs_mask)
-            loss_att = self.criterion(pred_pad, ys_out_pad_asr)
-
+            # [ADD] soft - hart tgt loss
+            loss_att = self.criterion_asr(pred_pad, ys_out_pad_asr, hyps_list, sentences_scores_list, soft_tgt_weight)
             acc = th_accuracy(
                 pred_pad.view(-1, self.odim),
                 ys_out_pad_asr,
@@ -464,7 +592,6 @@ class E2E(STInterface, torch.nn.Module):
             if not self.training:
                 ys_hat_asr = pred_pad.argmax(dim=-1)
                 cer, wer = self.error_calculator_asr(ys_hat_asr.cpu(), ys_pad.cpu())
-
         # CTC
         if self.mtlalpha > 0:
             batch_size = hs_pad.size(0)
@@ -505,7 +632,7 @@ class E2E(STInterface, torch.nn.Module):
         src_mask = (~make_pad_mask(ilens.tolist())).to(xs_zero_pad.device).unsqueeze(-2)
         hs_pad, hs_mask = self.encoder_mt(xs_zero_pad, src_mask)
         pred_pad, _ = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
-        loss = self.criterion(pred_pad, ys_out_pad)
+        loss = self.criterion_mt(pred_pad, ys_out_pad)
         acc = th_accuracy(
             pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
         )
@@ -706,7 +833,9 @@ class E2E(STInterface, torch.nn.Module):
         )
         return nbest_hyps
 
-    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
+    # [ADD] uttid_list
+    # def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src):
+    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_src, uttid_list):
         """E2E attention calculation.
 
         :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
@@ -719,7 +848,8 @@ class E2E(STInterface, torch.nn.Module):
         """
         self.eval()
         with torch.no_grad():
-            self.forward(xs_pad, ilens, ys_pad, ys_pad_src)
+            # [ADD] uttid_list
+            self.forward(xs_pad, ilens, ys_pad, ys_pad_src, uttid_list)
         ret = dict()

```
- espnet/nets/pytorch_backend/transformer/posterior_based_loss.py
	- espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py
```
+
+"""Copyed from Label smoothing module."""
+
+import torch
+from torch import nn
+
+from copy import deepcopy
+
+device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
+
+class PosteriorBasedLoss(nn.Module):
+
+    def __init__(
+        self,
+        size,
+        padding_idx,
+        smoothing,
+        normalize_length=False,
+        criterion=nn.KLDivLoss(reduction="none"),
+    ):
+        #super(LabelSmoothingLoss, self).__init__()
+        super(PosteriorBasedLoss, self).__init__()
+        self.criterion = criterion
+        self.padding_idx = padding_idx
+        self.confidence = 1.0 - smoothing
+        self.smoothing = smoothing
+        self.size = size
+        self.true_dist = None
+        self.normalize_length = normalize_length
+        self.softmax = torch.nn.Softmax(dim=1)
+
+    # [FIXME] TOO DIRTY
+    def padding_for_softloss(self, pred_pad, hyps_list, sentences_scores_list, vocab_size):
+        # [ADD] adjusted padding
+        pred_max_len = pred_pad.shape[1] # [64, 12, 7807]
+        scores_list_max_len = 0
+        one_best_list = []
+        for i in range(len(sentences_scores_list)):
+            scores_list_len = len(sentences_scores_list[i]) # token num exclude first sos of hyps_list
+            scores_list_max_len = max(scores_list_max_len, scores_list_len)
+            # one-best hyps
+            one_best = hyps_list[i][0]['yseq'][1:]
+            one_best_list.append(one_best)
+        # make zero_pad
+        zero_pad = [0] * vocab_size
+        zero_pad[0] = 1
+        pad_sentences_scores_list = []
+        pad_one_best_list = []
+        for i, scores_list in enumerate(sentences_scores_list):
+            # scores_list : one sentence
+            pad_one_sentence_scores_list = []
+            pad_one_best = one_best_list[i]
+            tokens_num = len(scores_list)
+            if tokens_num <= pred_max_len:
+                # zero_pad_list padding
+                pad_one_sentence_scores_list = scores_list
+                # softmax
+                pad_one_sentence_scores_list = self.softmax(torch.tensor(pad_one_sentence_scores_list))
+                pad_one_sentence_scores_list = pad_one_sentence_scores_list.tolist()
+                for j in range(pred_max_len - tokens_num):
+                    pad_one_sentence_scores_list.append(zero_pad)
+                    # padding
+                    # one_best
+                    pad_one_best.append(-1)
+            elif tokens_num > pred_max_len:
+                # reduce soft target tokens
+                # [FIXME] unefficient
+                pad_one_sentence_scores_list = scores_list[:pred_max_len]
+                # softmax
+                pad_one_sentence_scores_list = self.softmax(torch.tensor(pad_one_sentence_scores_list))
+                pad_one_sentence_scores_list = pad_one_sentence_scores_list.tolist()
+                # pad_one_best = pad_one_best[:pred_max_len]
+                pad_one_best = one_best_list[i][:pred_max_len]
+            pad_sentences_scores_list.append(pad_one_sentence_scores_list)
+            pad_one_best_list.append(pad_one_best)
+        # pad_sentences_scores_list = torch.tensor(pad_sentences_scores_list)
+        # to device
+        pad_sentences_scores_list = torch.tensor(pad_sentences_scores_list).to(device)
+        pad_one_best_list = torch.tensor(pad_one_best_list).to(device)
+        # one_best
+        return pad_one_best_list, pad_sentences_scores_list
+
+#            for j in range(pred_max_len): # make pred_max_len
+#                if j < len(scores_list):
+#                    # [FIXME] uneffective
+#                    pad_one_sentence_scores_list.append(scores_list[j])
+#                else:
+#                    pad_one_sentence_scores_list.append(zero_pad)
+#            pad_sentences_scores_list.append(pad_one_sentence_scores_list)
+#
+#        if pred_max_len >= scores_list_max_len:
+#            print(f'smaller: {pred_max_len} {scores_list_max_len}')
+#            # [COULDNT]
+#            # pad_sentences_scores_list = deepcopy(sentences_scores_list)
+#            pad_sentences_scores_list = []
+#
+#            for i, scores_list in enumerate(sentences_scores_list):
+#                # scores_list : one sentence
+#                pad_one_sentence_scores_list = []
+#                for j in range(pred_max_len): # make pred_max_len
+#                    if j < len(scores_list):
+#                        # [FIXME] uneffective
+#                        pad_one_sentence_scores_list.append(scores_list[j])
+#                    else:
+#                        pad_one_sentence_scores_list.append(zero_pad)
+#                pad_sentences_scores_list.append(pad_one_sentence_scores_list)
+#        [FIXME]
+#        elif pred_max_len < scores_list_max_len:
+#            print(f'bigger: {pred_max_len} {scores_list_max_len}')
+#            print('reduce long soft target part')
+#            pad_sentences_scores_list = []
+#            max_len = pred_max_len
+#
+#            for i, scores_list in enumerate(sentences_scores_list):
+#                # scores_list : one sentence
+#                pad_one_sentence_scores_list = []
+#                # [FIX]
+#                if len(scores_list) > pred_max_len:
+#                    for j in range(scores)list)
+#                elif len(scores_list) <= pred_max_len:
+#
+#                for j in range(pred_max_len): # make pred_max_len
+#                    if j < len(scores_list):
+#                        # [FIXME] uneffective
+#                        pad_one_sentence_scores_list.append(scores_list[j])
+#                    else:
+#                        pad_one_sentence_scores_list.append(zero_pad)
+#                pad_sentences_scores_list.append(pad_one_sentence_scores_list)
+
+    def forward(self, x, target, hyps_list, sentences_scores_list, soft_tgt_weight):
+        """Compute loss between x and target.
+
+        :param torch.Tensor x: prediction (batch, seqlen, class) (64, 12, 15210)
+        :param torch.Tensor target:
+            target signal masked with self.padding_id (batch, seqlen)
+        :return: scalar float value
+        :rtype torch.Tensor
+        """
+        assert x.size(2) == self.size
+        vocab_size = x.size(2)
+        pad_one_best_list, soft_target = self.padding_for_softloss(x, hyps_list, sentences_scores_list, vocab_size)
+
+        batch_size = x.size(0)
+        x = x.view(-1, self.size)
+        target = target.view(-1)
+        pad_one_best_list = pad_one_best_list.view(-1)
+        soft_target = soft_target.view(-1, self.size)
+        # x : [64, 16, 15211] -> [1024, 15211]
+        # target : [64, 16] -> [1024]
+        if soft_tgt_weight == 0:
+           # hard tgt
+            with torch.no_grad():
+                true_dist = x.clone()
+                true_dist.fill_(self.smoothing / (self.size - 1))
+                ignore = target == self.padding_idx  # (B,)
+                total = len(target) - ignore.sum().item()
+                target = target.masked_fill(ignore, 0)  # avoid -1 index
+                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
+            kl = self.criterion(torch.log_softmax(x, dim=1), true_dist)
+            denom = total if self.normalize_length else batch_size
+            loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
+            return loss
+
+        elif soft_tgt_weight == 1:
+            # soft tgt
+            # kl = self.criterion(torch.log_softmax(x, dim=1), torch.log_softmax(soft_target, dim=1))
+            # soft_target_softmax = []
+            # for soft in soft_target:
+            #     soft_target_softmax.append(torch.softmax(soft_target_softmax))
+            with torch.no_grad():
+                ignore = pad_one_best_list == self.padding_idx
+            kl = self.criterion(torch.log_softmax(x, dim=1), soft_target)
+            denom = total if self.normalize_length else batch_size
+            loss = kl.masked_fill(ignore.unsqueeze(1), 0).sum() / denom
+            return loss
+
+        # elif soft_tgt_weight > 0:
+        else:
+            with torch.no_grad():
+                true_dist = x.clone()
+                true_dist.fill_(self.smoothing / (self.size - 1))
+                ignore_hard = target == self.padding_idx  # (B,)
+                total = len(target) - ignore_hard.sum().item()
+                target = target.masked_fill(ignore_hard, 0)  # avoid -1 index
+                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
+            kl_hard = self.criterion(torch.log_softmax(x, dim=1), true_dist)
+
+            with torch.no_grad():
+                ignore_soft = pad_one_best_list == self.padding_idx
+            kl_soft = self.criterion(torch.log_softmax(x, dim=1), soft_target)
+
+        denom = total if self.normalize_length else batch_size
+        hard_loss = kl_hard.masked_fill(ignore_hard.unsqueeze(1), 0).sum() / denom
+        soft_loss = kl_soft.masked_fill(ignore_soft.unsqueeze(1), 0).sum() / denom
+
+        return (1 - soft_tgt_weight) * hard_loss + soft_tgt_weight * soft_loss

```
- espnet/st/pytorch_backend/st_pbl.py
	- espnet/st/pytorch_backend/st.py
```
@@ -37,7 +37,9 @@ from espnet.utils.dataset import ChainerDataLoader
 from espnet.utils.dataset import TransformDataset
 from espnet.utils.deterministic_utils import set_deterministic_pytorch
 from espnet.utils.dynamic_import import dynamic_import
-from espnet.utils.io_utils import LoadInputsAndTargets
+# [ADD]
+# from espnet.utils.io_utils import LoadInputsAndTargets
+from espnet.utils.io_utils_add_uttlist import LoadInputsAndTargets
 from espnet.utils.training.batchfy import make_batchset
 from espnet.utils.training.iterators import ShufflingEnabler
 from espnet.utils.training.tensorboard_logger import TensorboardLogger
@@ -88,7 +90,8 @@ class CustomConverter(ASRCustomConverter):
         """
         # batch should be located in list
         assert len(batch) == 1
-        xs, ys, ys_src = batch[0]
+        # [ADD] uttid_list
+        xs, ys, ys_src, uttid_list = batch[0]
 
         # get batch of lengths of input sequences
         ilens = np.array([x.shape[0] for x in xs])
@@ -111,7 +114,7 @@ class CustomConverter(ASRCustomConverter):
         else:
             ys_pad_src = None
 
-        return xs_pad, ilens, ys_pad, ys_pad_src
+        return xs_pad, ilens, ys_pad, ys_pad_src, uttid_list
 
 
 def train(args):
@@ -145,7 +148,7 @@ def train(args):
     assert isinstance(model, STInterface)
 
     if args.rnnlm is not None:
-        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
+        rnnlm_args = get_(args.rnnlm, args.rnnlm_conf)
         rnnlm = lm_pytorch.ClassifierWithState(
             lm_pytorch.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
         )
diff --git a/espnet/utils/io_utils.py b/espnet/utils/io_utils_add_uttlist.py
similarity index 99%
copy from espnet/utils/io_utils.py
copy to espnet/utils/io_utils_add_uttlist.py

```
- espnet/utils/io_utils_add_uttlist.py
	- espnet/utils/io_utils.py
```
@@ -186,9 +186,12 @@ class LoadInputsAndTargets(object):
                     return_batch[x_name] = self.preprocessing(
                         return_batch[x_name], uttid_list, **self.preprocess_args
                     )
-
         # Doesn't return the names now.
-        return tuple(return_batch.values())
+
+        # [ADD] return uttid_list also
+        # [BEFORE] return tuple(return_batch.values())
+        utt_ll = [uttid_list]
+        return tuple(list(return_batch.values()) + utt_ll)
 
     def _create_batch_asr(self, x_feats_dict, y_feats_dict, uttid_list):
         """Create a OrderedDict for the mini-batch
```
