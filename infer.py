"""Inference script for the cleaned baseline pipeline."""
import argparse
import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

from abaw.abaw_dataset import HumeDatasetEval
from abaw.utils import setup_system
from abaw.evaluate import predict
from abaw.model_ablation import ModelAblation


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
    verbose: bool = True
    gpu_ids: tuple = (0,)
    task: str = 'text+vit+audio'
    batch_size_eval: int = 32
    mtl: bool = False
    modality_dropout_p: float = 0.0
    use_gate_intensity: bool = False
    data_folder: str = './data/test/'
    model_path: str = './hume_model'
    num_workers: int = 0 if os.name == 'nt' else 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn_benchmark: bool = True
    cudnn_deterministic: bool = False


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference for baseline model.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth). If omitted, auto-find newest weights_end.pth in hume_model.')
    parser.add_argument('--output-csv', type=str, default='submissions/pred2026_CA_all.csv',
                        help='Prediction CSV output path.')
    parser.add_argument('--label-file', type=str, default='data/test/test_split.csv',
                        help='CSV file listing test sample IDs.')
    parser.add_argument('--data-folder', type=str, default='./data/test/',
                        help='Test data folder containing vit/googlevit, wav2vec2, and text subfolders.')
    parser.add_argument('--batch-size-eval', type=int, default=32, help='Inference batch size.')
    parser.add_argument('--audio-model', default=AUDIO_DEFAULT, help='Audio encoder model path or HF id.')
    parser.add_argument('--text-model', default=TEXT_DEFAULT, help='Text encoder model path or HF id.')
    return parser.parse_args()


def find_latest_checkpoint(model_root='hume_model'):
    root = Path(model_root)
    if not root.exists():
        return None
    candidates = list(root.glob('**/weights_end.pth'))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


config = Config()

if __name__ == '__main__':
    args = parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint(config.model_path)
    if checkpoint is None:
        raise FileNotFoundError('No checkpoint found. Please pass --checkpoint or place weights_end.pth under ./hume_model.')
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    output_csv = args.output_csv
    label_file = args.label_file
    config.data_folder = args.data_folder
    config.batch_size_eval = args.batch_size_eval
    config.model = (
        'linear',
        resolve_model_name(args.audio_model, 'audio'),
        resolve_model_name(args.text_model, 'text'),
    )

    setup_system(config.seed, config.cudnn_benchmark, config.cudnn_deterministic)
    print(f"Checkpoint : {checkpoint}")
    print(f"Output     : {output_csv}")
    print(f"Device     : {config.device}")
    print(f"Audio model: {config.model[1]}")
    print(f"Text model : {config.model[2]}")

    model = ModelAblation(
        config.model,
        config.task,
        modality_dropout_p=0.0,
        use_projection=False,
        use_vision_temporal=False,
        use_fusion_self_attn=False,
        use_sigmoid=False,
        use_gate_intensity=False,
        freeze_encoders=False,
    ).to(config.device)

    state = torch.load(checkpoint, map_location=config.device)
    model.load_state_dict(state)
    model.eval()
    print('Weights loaded.')

    test_dataset = HumeDatasetEval(
        data_folder=config.data_folder,
        label_file=label_file,
        model=config.model,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size_eval,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=test_dataset.collate_fn,
    )
    print(f"Test samples: {len(test_dataset)}")

    preds, _, filenames = predict(config, model, test_loader)
    preds_np = preds.numpy()

    cols = ['Filename', 'Admiration', 'Amusement', 'Determination',
            'Empathic Pain', 'Excitement', 'Joy']
    df = pd.DataFrame(
        np.hstack([np.array(filenames).reshape(-1, 1), preds_np]),
        columns=cols,
    )
    df['Filename'] = df['Filename'].astype(int)
    df = df.sort_values(by='Filename').reset_index(drop=True)

    os.makedirs('submissions', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\\nSaved {len(df)} rows  ->  {output_csv}")
