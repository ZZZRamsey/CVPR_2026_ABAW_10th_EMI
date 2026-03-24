# :trophy: Champion Solution for the EMI Track (CVPR 2026 ABAW Challenge) from USTC 

Entrypoints:
- train.py: training on train split
- infer.py: inference with optional checkpoint auto-discovery

Core runtime modules are kept in abaw/.

## 1. Setup

Download pretrained model from huggingface and put it into "./pretrained".
```bash
pretrained/vit-base-patch16-224-in21k
pretrained/whisper-large-v3-turbo
pretrained/wav2vec2-large-robust-12-ft-emotion-msp-dim
```

## 2. Required Data Layout

Training side:
- data/train_split.csv
- data/valid_split.csv
- data/wav2vec2/*.pkl
- data/vit/*.pkl or data/googlevit/*.pkl
- data/text/*.txt

Inference side:
- data/test/test_split.csv
- data/test/wav2vec2/*.pkl
- data/test/vit/*.pkl or data/test/googlevit/*.pkl
- data/test/text/*.txt

## 3. Preprocessing Commands

If your features already exist, skip this section.

```bash
python tools/whisper_transcribe.py
python tools/extract_google.py
python tools/prepare_test_data.py
```

## 4. Training Command

Train split:

```bash
python train.py \
  --train-csv data/train_split.csv \
  --val-csv data/valid_split.csv \
  --epochs 50 \
  --batch-size 32 \
  --batch-size-eval 32 \
  --patience 5 \
  --exp-name Baseline_CA_MSE
```


Outputs are under hume_model/<timestamp>_<exp-name>/.
Important checkpoints:
- weights_best.pth
- weights_end.pth

## 5. Inference Commands

With explicit checkpoint:

```bash
python infer.py \
  --checkpoint hume_model/your_run/weights_end.pth \
  --label-file data/test/test_split.csv \
  --data-folder ./data/test/ \
  --batch-size-eval 32 \
  --output-csv submissions/pred2026_CA_all.csv
```

Auto pick latest checkpoint:

```bash
python infer.py \
  --label-file data/test/test_split.csv \
  --data-folder ./data/test/ \
  --batch-size-eval 32 \
  --output-csv submissions/pred2026_CA.csv
```
