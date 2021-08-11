# ASR Posterior-based Loss for Multi-task Speech Translation

- [Paper(INTERSPEECH2021)](link)

## Dependencies
- We will need Kaldi and Python environment of ESPnet (We used venv). 
- `${HOME}` is your working directory. 

```
Python >= 3.7
torch==1.6.0
```

## 1. Preprocess for ST
```
cd ${HOME}/egs/fisher_callhome_spanish/st1/
```
- Set `stage=0` and `stop_stage=3`
```
bash run-proposed.sh
```

## 2. Training Pre-trained ASR
```
cd ${HOME}/egs/fisher_callhome_spanish/asr1b/
```
- We use ST dict for pre-training ASR. 

```bash:run-proposed.sh
 dict=../st1/data/lang_1spm/train_sp.en_bpe1000_units_lc.rm.txt
 nlsyms=../st1/data/lang_1spm/train_sp.en_non_lang_syms_lc.rm.txt
 bpemodel=../st1/data/lang_1spm/train_sp.en_bpe1000_lc.rm
```
- Train ASR model. (stage0-stage5)
```
bash run-proposed.sh
```

## 3. Generating Pre-decoding soft labels from pre-trained ASR
- We added configuration. 
- `pre_decoding_recog_model=<PRETRAIN_ASR_MODEL_PATH>`
- `pre_decoding_dir=<SOFT_LABELS_SAVING_DIR_PATH>`

```bash:run-proposed.sh
# [ADD] configuration
data=data
pre_decoding_recog_model=../asr1b/exp/pretrain-asr-for-pbl/results/model.val1.avg.best
pre_decoding_dir=../asr1b/exp/pretrain-asr-for-pbl/pre_decoding
model_module="espnet.nets.pytorch_backend.e2e_st_transformer_asrpbl:E2E"
epochs=30
asr_weight=0.4
lsm_weight_st=0.1
lsm_weight_asr=0.1
soft_tgt_weight=0.5
```
- Instead of `stage 3:LM Preparation`, we added `stage 3: Pre-Decoding dev train`
```bash:run-proposed.sh
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Pre-Decoding dev train"
```
# 4. Training ST model (stage3-stage5). 
```
bash run-proposed.sh
```
## Changed logs
Our changed logs are on [changed_log.md](https://github.com/ahclab/st-asrpbl/blob/master/changed_log.md).

## References and acknowledgement

This repository is a clone of [ESPnet](https://github.com/espnet/espnet). You should consider citing their papers as well if you use this code. 
 
