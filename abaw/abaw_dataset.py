import cv2
import imageio_ffmpeg
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
import pickle
import timm
import math
from fractions import Fraction
from transformers import AutoProcessor, Wav2Vec2FeatureExtractor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import soundfile as sf
import abaw.utils
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence
from pathlib import Path
from transformers import AutoTokenizer
import audiomentations
cv2.setNumThreads(2)

EMOTION_COLS = ['Admiration', 'Amusement', 'Determination', 'Empathic Pain', 'Excitement', 'Joy']


def compute_annotator_count(row):
    """Infer number of annotators N from label fractions (labels are k/N)."""
    nonzero_vals = [row[c] for c in EMOTION_COLS if row[c] > 0]
    if not nonzero_vals:
        return 1
    max_denom = 1
    for v in nonzero_vals:
        f = Fraction(v).limit_denominator(200)
        max_denom = max(max_denom, f.denominator)
    return max_denom


class HumeDatasetTrain(Dataset, abaw.utils.AverageMeter):

    def __init__(self, data_folder, label_file=None, model=None, use_annotation_weight=False):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]
        self.use_annotation_weight = use_annotation_weight

        # Precompute annotation weights
        if self.use_annotation_weight:
            self.annotation_weights = []
            for i in range(len(self.label_file)):
                row = self.label_file.iloc[i]
                N = compute_annotator_count(row)
                self.annotation_weights.append(math.sqrt(N))
            # Normalize so mean weight = 1
            mean_w = np.mean(self.annotation_weights)
            self.annotation_weights = [w / mean_w for w in self.annotation_weights]
            print(f"Annotation weights: min={min(self.annotation_weights):.3f}, "
                  f"max={max(self.annotation_weights):.3f}, "
                  f"mean={np.mean(self.annotation_weights):.3f}")

        self.wave_transforms = audiomentations.Compose([
                            audiomentations.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.15, p=0.5),
                            audiomentations.TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5),
                            audiomentations.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
                            audiomentations.Shift(min_shift=-0.1, max_shift=0.1, p=0.5),
                            ])
        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])

        if self.audio_model != 'linear':
            if "pretrained" in self.audio_model:
                self.processor = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model)
            self.processor_vision = AutoProcessor.from_pretrained('pretrained/vit-base-patch16-224-in21k')
        self.processor_text = AutoTokenizer.from_pretrained(self.text_model)

        # Auto-detect vit feature dir and dim
        if os.path.exists(f"{self.data_folder}googlevit") and len(os.listdir(f"{self.data_folder}googlevit")) > 0:
            self.vit_dir = "googlevit"
        else:
            self.vit_dir = "vit"
        _first = sorted(os.listdir(f"{self.data_folder}{self.vit_dir}"))[0]
        with open(f"{self.data_folder}{self.vit_dir}/{_first}", 'rb') as _f:
            _s = pickle.load(_f)
            self.vit_feat_dim = _s.shape[-1] if hasattr(_s, 'shape') else 768

    def __getitem__(self, index):
        row = self.label_file.iloc[index]
        vision_missing = 0
        text_missing = 0

        if self.vision_model == 'linear':
            try:
                vit_file_path = f"{self.data_folder}{self.vit_dir}/{str(int(row['Filename'])).zfill(5)}.pkl"
                with open(vit_file_path, 'rb') as file:
                    data = pickle.load(file)
                    if isinstance(data, torch.Tensor):
                        tensor = data
                    else:
                        tensor = torch.tensor(data)
                    length = tensor.size(0)
                    max_length = 400
                    if length < max_length:
                        pad_size = (0, 0, 0, max_length - length)
                        vision = torch.nn.functional.pad(tensor, pad_size)
                    else:
                        vision = tensor[:max_length]
                        length = 400
            except Exception as e:
                print(e)
                vision = torch.zeros(400, self.vit_feat_dim)
                length = 1
                vision_missing = 1
        else:
            vision = self.process_images(index)

        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])

        labels = torch.tensor(
            row[EMOTION_COLS].values,
            dtype=torch.float)

        text, text_missing = self.process_text(row['Filename'])

        # Annotation weight
        if self.use_annotation_weight:
            weight = torch.tensor(self.annotation_weights[index], dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)

        return audio, vision, torch.tensor(length).long(), text, labels, self.avg, vision_missing, text_missing, weight

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            images = []
            while len(images) < 1:
                black_img = Image.new('RGB', (224, 224))
                images.append(self.transform(image=np.array(black_img))['image'])
            return torch.stack(images)
        except Exception as e:
            images = []
            print(e)
            while len(images) < 1:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
            audio_data = audio_data.astype(np.float32)
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros(120*16000+1, dtype=np.float32)
        self.update(1 - len(audio_data[:12*sr])/len(audio_data))
        if audio_data.ndim == 1:
            audio_data = np.expand_dims(audio_data, axis=0)
        elif audio_data.shape[1] == 1:
            audio_data = audio_data.T
        audio_data = audio_data.squeeze(axis=0)
        return audio_data[:12*sr]

    def process_text(self, filename):
        text_file_path = f"{self.data_folder}text/{str(int(filename)).zfill(5)}.txt"
        try:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            if len(text) == 0:
                return "", 1
            return text, 0
        except FileNotFoundError:
            return "", 1

    def __len__(self):
        return len(self.label_file)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, labels_data, avg, vision_missing, text_missing, weights = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=12*16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)

        labels_stacked = torch.stack(labels_data)
        encoded_text = self.processor_text(
            text_data,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        vision_missing_t = torch.tensor(vision_missing, dtype=torch.long)
        text_missing_t = torch.tensor(text_missing, dtype=torch.long)
        weights_t = torch.stack(weights)
        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, labels_stacked, np.mean(avg), vision_missing_t, text_missing_t, weights_t


class HumeDatasetEval(Dataset):

    def __init__(self, data_folder, label_file=None, model=None):
        super().__init__()
        self.data_folder = data_folder
        self.label_file = pd.read_csv(label_file)
        self.vision_model = model[0]
        self.audio_model = model[1]
        self.text_model = model[2]

        if self.vision_model != 'linear':
            self.data_config = timm.data.resolve_model_data_config(self.vision_model)
            self.transform = A.Compose([
                A.Resize(height=self.data_config['input_size'][1], width=self.data_config['input_size'][2]),
                A.Normalize(mean=self.data_config['mean'], std=self.data_config['std']),
                ToTensorV2(),
            ])
        if self.audio_model != 'linear':
            if "pretrained" in self.audio_model:
                self.processor = AutoProcessor.from_pretrained(self.audio_model)
            else:
                self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.audio_model)
            self.processor_vision = AutoProcessor.from_pretrained('pretrained/vit-base-patch16-224-in21k')
        self.processor_text = AutoTokenizer.from_pretrained(self.text_model)

        # Auto-detect vit feature dir and dim
        if os.path.exists(f"{self.data_folder}googlevit") and len(os.listdir(f"{self.data_folder}googlevit")) > 0:
            self.vit_dir = "googlevit"
        else:
            self.vit_dir = "vit"
        _first = sorted(os.listdir(f"{self.data_folder}{self.vit_dir}"))[0]
        with open(f"{self.data_folder}{self.vit_dir}/{_first}", 'rb') as _f:
            _s = pickle.load(_f)
            self.vit_feat_dim = _s.shape[-1] if hasattr(_s, 'shape') else 768

    def __getitem__(self, index):
        row = self.label_file.iloc[index]
        vision_missing = 0
        text_missing = 0

        if self.vision_model == 'linear':
            try:
                vit_file_path = f"{self.data_folder}{self.vit_dir}/{str(int(row['Filename'])).zfill(5)}.pkl"
                with open(vit_file_path, 'rb') as file:
                    data = pickle.load(file)
                    if isinstance(data, torch.Tensor):
                        tensor = data
                    else:
                        tensor = torch.tensor(data)
                    length = tensor.size(0)
                    max_length = 400
                    if length < max_length:
                        pad_size = (0, 0, 0, max_length - length)
                        vision = torch.nn.functional.pad(tensor, pad_size)
                    else:
                        vision = tensor[:max_length]
                        length = 400

            except Exception as e:
                vision = torch.zeros(400, self.vit_feat_dim)
                length = 1
                vision_missing = 1
        else:
            vision = self.process_images(index)

        if self.audio_model == 'linear':
            wav2vec2_file_path = f"{self.data_folder}wav2vec2/{str(int(row['Filename'])).zfill(5)}.pkl"
            with open(wav2vec2_file_path, 'rb') as file:
                audio = torch.mean(torch.tensor(pickle.load(file)), dim=0)
        else:
            audio = self.process_audio(row['Filename'])
        labels = torch.tensor(
            row[EMOTION_COLS].values,
            dtype=torch.float)

        text, text_missing = self.process_text(row['Filename'])

        return audio, vision, torch.tensor(length).long(), text, labels, int(row['Filename']), vision_missing, text_missing

    def process_images(self, index):
        try:
            img_folder_path = f"{self.data_folder}face_images/{str(int(index)).zfill(5)}/"
            img_files = sorted(os.listdir(img_folder_path), key=lambda x: x.zfill(15))
            images = []
            while len(images) < 1:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            return torch.stack(images)
        except:
            images = []
            while len(images) < 1:
                black_img = Image.new('RGB', (160, 160))
                images.append(self.transform(image=np.array(black_img))['image'])
            print(f"No image found for index: {index}")
            return torch.stack(images)

    def process_audio(self, filename):
        audio_file_path = f"{self.data_folder}audio/{str(int(filename)).zfill(5)}.mp3"
        try:
            audio_data, sr = sf.read(audio_file_path)
            if sr != 16000:
                print(audio_file_path)
                raise ValueError
        except Exception as e:
            print(f"Error processing audio file {audio_file_path}: {e}")
            audio_data = np.zeros((128,), dtype=np.float32)
            sr = 1
        return audio_data[:12*sr]

    def process_text(self, filename):
        text_file_path = f"{self.data_folder}text/{str(int(filename)).zfill(5)}.txt"
        try:
            with open(text_file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
            if len(text) == 0:
                return "", 1
            return text, 0
        except FileNotFoundError:
            return "", 1

    def __len__(self):
        return len(self.label_file)

    def collate_fn(self, batch):
        audio_data, vision_data, max_length, text_data, labels_data, avg, vision_missing, text_missing = zip(*batch)
        audio_data_padded = self.processor(audio_data, padding=True, sampling_rate=16000, return_tensors="pt", truncation=True, max_length=12*16000, return_attention_mask=True)
        lengths, permutation = audio_data_padded['attention_mask'].sum(axis=1).sort(descending=True)

        labels_stacked = torch.stack(labels_data)
        encoded_text = self.processor_text(
            text_data,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        vision_missing_t = torch.tensor(vision_missing, dtype=torch.long)
        text_missing_t = torch.tensor(text_missing, dtype=torch.long)
        return audio_data_padded, torch.stack(vision_data), torch.stack(max_length), encoded_text, labels_stacked, avg, vision_missing_t, text_missing_t
