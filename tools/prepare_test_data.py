"""Prepare test data: transcribe audio with whisper and extract googlevit features."""
import os
import torch
import pickle
from tqdm import tqdm


def transcribe_test_audio():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    model_id = "pretrained/whisper-large-v3-turbo"
    device = "cuda"
    torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_dir = "data/test/audio/"
    output_dir = "data/test/text/"
    os.makedirs(output_dir, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.mp3')])

    todo = []
    for f in audio_files:
        txt_path = os.path.join(output_dir, os.path.splitext(f)[0] + '.txt')
        if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
            todo.append(f)

    print(f"Transcription: {len(audio_files)} total, {len(todo)} todo")

    for filename in tqdm(todo, desc="Transcribing"):
        audio_path = os.path.join(audio_dir, filename)
        try:
            result = pipe(audio_path, return_timestamps=True, generate_kwargs={"language": "english"})
            text = result["text"].strip()
        except Exception as e:
            print(f"Error on {filename}: {e}")
            text = ""

        text_filename = os.path.splitext(filename)[0] + '.txt'
        text_path = os.path.join(output_dir, text_filename)
        with open(text_path, 'w') as f:
            f.write(text)

    print(f"Transcription done! Files: {len(os.listdir(output_dir))}")

    del model, pipe
    torch.cuda.empty_cache()


def extract_googlevit():
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    input_folder = 'data/test/face_images'
    output_folder = 'data/test/googlevit'
    os.makedirs(output_folder, exist_ok=True)

    device = 'cuda'
    processor = AutoImageProcessor.from_pretrained('pretrained/vit-base-patch16-224-in21k')
    model = AutoModel.from_pretrained('pretrained/vit-base-patch16-224-in21k').to(device)
    model.eval()
    model.half()

    BATCH_SIZE = 64

    scenes = sorted(os.listdir(input_folder))
    existing = set(os.listdir(output_folder))
    todo = [s for s in scenes if os.path.isdir(os.path.join(input_folder, s)) and f'{s}.pkl' not in existing]

    print(f"GoogleViT extraction: {len(scenes)} total scenes, {len(todo)} todo")

    for scene in tqdm(todo, desc='Extracting GoogleViT'):
        scene_path = os.path.join(input_folder, scene)
        if not os.path.isdir(scene_path):
            continue

        image_files = sorted([f for f in os.listdir(scene_path) if f.endswith('.jpg')],
                             key=lambda x: int(os.path.splitext(x)[0]))

        if not image_files:
            scene_tensor = torch.zeros(1, 768)
            save_path = os.path.join(output_folder, f'{scene}.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(scene_tensor, f)
            continue

        features = []
        for i in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[i:i+BATCH_SIZE]
            images = [Image.open(os.path.join(scene_path, f)).convert('RGB') for f in batch_files]
            inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = model(**inputs)
                batch_features = outputs.last_hidden_state.mean(dim=1)

            features.append(batch_features.float().cpu())

        scene_tensor = torch.cat(features, dim=0)
        save_path = os.path.join(output_folder, f'{scene}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(scene_tensor, f)

    print(f"GoogleViT extraction done! Files: {len(os.listdir(output_folder))}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    print("=" * 60)
    print("Step 1: Whisper Transcription")
    print("=" * 60)
    transcribe_test_audio()

    print("\n" + "=" * 60)
    print("Step 2: GoogleViT Feature Extraction")
    print("=" * 60)
    extract_googlevit()

    print("\nAll test data preparation complete!")
