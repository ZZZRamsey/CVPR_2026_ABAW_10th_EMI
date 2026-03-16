import argparse
import os
import pickle

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification


def numeric_stem_sort_key(name: str):
    stem, _ = os.path.splitext(name)
    try:
        return int(stem)
    except ValueError:
        return stem


def build_model(model_path: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    if device.startswith('cuda'):
        model.half()
    return processor, model


def batch_extract_features(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if getattr(outputs, 'last_hidden_state', None) is not None:
        return outputs.last_hidden_state.mean(dim=1)

    hidden_states = getattr(outputs, 'hidden_states', None)
    if hidden_states is not None and len(hidden_states) > 0:
        return hidden_states[-1].mean(dim=1)

    if getattr(outputs, 'pooler_output', None) is not None:
        return outputs.pooler_output

    if getattr(outputs, 'logits', None) is not None:
        return outputs.logits

    raise RuntimeError('No usable feature tensor found in model outputs')


def main():
    parser = argparse.ArgumentParser(description='Extract frame-wise face features from scene folders')
    parser.add_argument('--input', default='data/face_images', help='Input folder containing scene subfolders')
    parser.add_argument('--output', required=True, help='Output folder for scene pkl files')
    parser.add_argument('--model', required=True, help='HuggingFace model path/name')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for image encoding')
    parser.add_argument('--skip-existing', action='store_true', help='Skip scene if output pkl exists')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)

    print(f'[Extractor] model={args.model}')
    print(f'[Extractor] input={args.input}')
    print(f'[Extractor] output={args.output}')
    print(f'[Extractor] device={device}, batch_size={args.batch_size}')

    processor, model = build_model(args.model, device)

    scenes = sorted([d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))])
    for scene in tqdm(scenes, desc='Scenes'):
        scene_dir = os.path.join(args.input, scene)
        out_path = os.path.join(args.output, f'{scene}.pkl')

        if args.skip_existing and os.path.exists(out_path):
            continue

        image_files = sorted(
            [f for f in os.listdir(scene_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=numeric_stem_sort_key,
        )
        if not image_files:
            continue

        feats = []
        for i in range(0, len(image_files), args.batch_size):
            batch_files = image_files[i:i + args.batch_size]
            images = [Image.open(os.path.join(scene_dir, f)).convert('RGB') for f in batch_files]
            inputs = processor(images=images, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            if device.startswith('cuda'):
                with torch.amp.autocast('cuda'):
                    batch_feats = batch_extract_features(model, inputs)
            else:
                batch_feats = batch_extract_features(model, inputs)

            feats.append(batch_feats.float().cpu())

        scene_tensor = torch.cat(feats, dim=0)
        with open(out_path, 'wb') as f:
            pickle.dump(scene_tensor, f)

    print('[Extractor] Done.')


if __name__ == '__main__':
    main()
