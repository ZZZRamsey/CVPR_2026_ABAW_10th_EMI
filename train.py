"""Baseline training script (train split)."""
import argparse
import os
import time
import math
import shutil
import sys
import torch
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from abaw.abaw_dataset import HumeDatasetEval, HumeDatasetTrain
from abaw.utils import setup_system, Logger
from abaw.trainer import train
from abaw.evaluate import evaluate
from abaw.loss import MSE
from abaw.model_ablation import ModelAblation
from transformers import get_cosine_schedule_with_warmup


AUDIO_DEFAULT = 'pretrained/wav2vec2-large-robust-12-ft-emotion-msp-dim'
TEXT_DEFAULT = 'pretrained/gte-en-mlm-base'


def resolve_model_name(name: str, kind: str) -> str:
    if os.path.exists(name):
        return name
    if kind == 'audio' and name == AUDIO_DEFAULT:
        return 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    if kind == 'text' and name == TEXT_DEFAULT:
        return 'distilbert-base-uncased'
    return name


@dataclass
class Config:
    model: tuple = ('linear', AUDIO_DEFAULT, TEXT_DEFAULT)
    mixed_precision: bool = True
    seed: int = 3407
    epochs: int = 50
    batch_size: int = 32
    verbose: bool = True
    gpu_ids: tuple = (0,)
    task: str = 'text+vit+audio'
    batch_size_eval: int = 32
    eval_every_n_epoch: int = 1
    clip_grad: float = 100.0
    decay_exclue_bias: bool = False
    grad_checkpointing: bool = False
    gradient_accumulation: int = 1
    loss: str = 'MSE'
    mtl: bool = False
    lr: float = 1e-4
    lr_encoder: float = None
    scheduler: str = 'cosine'
    warmup_epochs: int = 1
    lr_end: float = 1e-4
    modality_dropout_p: float = 0.1
    use_projection: bool = False
    use_vision_temporal: bool = False
    use_fusion_self_attn: bool = False
    use_sigmoid: bool = False
    use_gate_intensity: bool = False
    freeze_encoders: bool = False
    use_annotation_weight: bool = False
    data_folder: str = './data/'
    model_path: str = './hume_model'
    checkpoint_start = None
    num_workers: int = 0 if os.name == 'nt' else 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False
    patience: int = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline model on train split.')
    parser.add_argument('--data-folder', default='./data/', help='Data root folder.')
    parser.add_argument('--train-csv', default='data/train_split.csv', help='Training csv path.')
    parser.add_argument('--val-csv', default='data/valid_split.csv', help='Validation csv path.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size.')
    parser.add_argument('--batch-size-eval', type=int, default=32, help='Evaluation batch size.')
    parser.add_argument('--num-workers', type=int, default=None, help='Dataloader workers.')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience.')
    parser.add_argument('--model-path', default='./hume_model', help='Checkpoint output directory.')
    parser.add_argument('--exp-name', default='Baseline_CA_MSE', help='Experiment name suffix in output folder.')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')
    parser.add_argument('--audio-model', default=AUDIO_DEFAULT, help='Audio encoder model path or HF id.')
    parser.add_argument('--text-model', default=TEXT_DEFAULT, help='Text encoder model path or HF id.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = Config()
    config.data_folder = args.data_folder
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.batch_size_eval = args.batch_size_eval
    config.patience = args.patience
    config.model_path = args.model_path
    config.seed = args.seed
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.model = (
        'linear',
        resolve_model_name(args.audio_model, 'audio'),
        resolve_model_name(args.text_model, 'text'),
    )

    model_path = f"{config.model_path}/{time.strftime('%Y%m%d_%H%M%S')}_{args.exp_name}"
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.basename(__file__), f"{model_path}/train.py")
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))

    print(f"Experiment : {args.exp_name}")
    print(f"Model path : {model_path}")
    print(f"Audio model: {config.model[1]}")
    print(f"Text model : {config.model[2]}")

    setup_system(config.seed, config.cudnn_benchmark, config.cudnn_deterministic)

    model = ModelAblation(
        config.model,
        config.task,
        modality_dropout_p=config.modality_dropout_p,
        use_projection=config.use_projection,
        use_vision_temporal=config.use_vision_temporal,
        use_fusion_self_attn=config.use_fusion_self_attn,
        use_sigmoid=config.use_sigmoid,
        use_gate_intensity=config.use_gate_intensity,
        freeze_encoders=config.freeze_encoders,
    ).to(config.device)

    train_dataset = HumeDatasetTrain(
        data_folder=config.data_folder,
        label_file=args.train_csv,
        model=config.model,
        use_annotation_weight=config.use_annotation_weight,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataset = HumeDatasetEval(
        data_folder=config.data_folder,
        label_file=args.val_csv,
        model=config.model,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=eval_dataset.collate_fn,
    )
    print(f"Train: {len(train_dataset)}  Val: {len(eval_dataset)}")

    loss_fn = MSE()
    print(f"Loss={config.loss}")
    scaler = GradScaler(init_scale=2.0 ** 10) if config.mixed_precision else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    print(f"Optim=AdamW lr={config.lr}")

    total_steps = math.floor(len(train_loader) * config.epochs / config.gradient_accumulation)
    warmup_steps = len(train_loader) * config.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Epochs={config.epochs}  patience={config.patience}\\n")

    best_score, patience_ctr = 0.0, 0
    for epoch in range(1, config.epochs + 1):
        print(f"\\n{'-' * 30}[Epoch {epoch}]{'-' * 30}")
        train_loss = train(config, model, train_loader, loss_fn, optimizer, scheduler, scaler)
        print(f"Epoch {epoch}  Train Loss={train_loss:.4f}  Lr={optimizer.param_groups[0]['lr']:.6f}")

        corr, _, _ = evaluate(config, model, eval_loader)
        score = float(corr)
        if math.isnan(score):
            score = 0.0

        if score > best_score:
            best_score = score
            patience_ctr = 0
            torch.save(model.state_dict(), f"{model_path}/weights_best.pth")
        else:
            patience_ctr += 1

        print(f"Epoch {epoch}  Eval Pearson={score:.4f}  Best={best_score:.4f}  Patience={patience_ctr}/{config.patience}")
        torch.save(model.state_dict(), f"{model_path}/weights_e{epoch}_{score:.4f}.pth")
        if patience_ctr >= config.patience:
            print(f"\\n*** Early stopping at epoch {epoch} ***")
            break

    torch.save(model.state_dict(), f"{model_path}/weights_end.pth")
    print(f"\\nTraining complete.  Best Eval Pearson = {best_score:.4f}")
