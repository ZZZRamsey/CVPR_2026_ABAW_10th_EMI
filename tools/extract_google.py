import os
import torch
import pickle
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

# Paths
input_folder = 'data/face_images'
output_folder = 'data/googlevit'

# Create output directory if not exists
os.makedirs(output_folder, exist_ok=True)

# Load ViT model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
processor = AutoImageProcessor.from_pretrained('pretrained/vit-base-patch16-224-in21k')
model = AutoModel.from_pretrained('pretrained/vit-base-patch16-224-in21k').to(device)
model.eval()
model.half()  # FP16 for faster inference

BATCH_SIZE = 64

# Process each scene
scenes = sorted(os.listdir(input_folder))
for scene in tqdm(scenes, desc='Scenes'):
    scene_path = os.path.join(input_folder, scene)
    if not os.path.isdir(scene_path):
        continue

    # Skip already processed scenes
    save_path = os.path.join(output_folder, f'{scene}.pkl')
    if os.path.exists(save_path):
        continue

    # Sort files numerically (0.jpg, 1.jpg, etc.)
    image_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.jpg')],
                         key=lambda x: int(os.path.splitext(x)[0]))

    features = []
    # Process in batches for speed
    for i in range(0, len(image_files), BATCH_SIZE):
        batch_files = image_files[i:i+BATCH_SIZE]
        images = [Image.open(os.path.join(scene_path, f)).convert('RGB') for f in batch_files]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model(**inputs)
            batch_features = outputs.last_hidden_state.mean(dim=1)  # [batch, feature_size]

        features.append(batch_features.float().cpu())

    # Stack features [sequence_length, feature_size]
    scene_tensor = torch.cat(features, dim=0)

    with open(save_path, 'wb') as f:
        pickle.dump(scene_tensor, f)

print("Encoding completed successfully.")
