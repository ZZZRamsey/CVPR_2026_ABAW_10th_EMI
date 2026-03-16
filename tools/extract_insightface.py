import argparse
import os
import pickle
from typing import List, Optional

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sort_key(name: str):
    stem = os.path.splitext(name)[0]
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _list_scenes(input_dir: str) -> List[str]:
    scenes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    return sorted(scenes, key=_sort_key)


def _list_images(scene_dir: str) -> List[str]:
    files = [
        f
        for f in os.listdir(scene_dir)
        if os.path.isfile(os.path.join(scene_dir, f))
        and os.path.splitext(f.lower())[1] in IMAGE_EXTS
    ]
    return sorted(files, key=_sort_key)


def _pick_largest_face(faces):
    if not faces:
        return None
    best = None
    best_area = -1.0
    for face in faces:
        bbox = face.bbox
        area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        if area > best_area:
            best_area = area
            best = face
    return best


def _build_app(model_name: str, model_root: str, det_size: int, use_cuda: bool) -> FaceAnalysis:
    providers = ["CPUExecutionProvider"]
    if use_cuda:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    try:
        app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        app.prepare(ctx_id=0 if use_cuda else -1, det_size=(det_size, det_size))
        return app
    except Exception:
        # Fall back to CPU if CUDA provider is unavailable.
        app = FaceAnalysis(name=model_name, root=model_root, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        return app


def extract(
    input_dir: str,
    output_dir: str,
    app: FaceAnalysis,
    fallback_dim: int,
    overwrite: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    scenes = _list_scenes(input_dir)
    if not scenes:
        raise RuntimeError(f"No scene folders found in: {input_dir}")

    total_images = 0
    miss_faces = 0
    saved_files = 0
    inferred_dim: Optional[int] = None

    for scene in tqdm(scenes, desc="Scenes"):
        scene_dir = os.path.join(input_dir, scene)
        image_files = _list_images(scene_dir)
        if not image_files:
            continue

        save_path = os.path.join(output_dir, f"{scene}.pkl")
        if (not overwrite) and os.path.exists(save_path):
            continue

        features = []
        for image_file in image_files:
            total_images += 1
            image_path = os.path.join(scene_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                miss_faces += 1
                dim = inferred_dim if inferred_dim is not None else fallback_dim
                features.append(torch.zeros(dim, dtype=torch.float32))
                continue

            faces = app.get(image)
            face = _pick_largest_face(faces)
            if face is None or face.embedding is None:
                miss_faces += 1
                dim = inferred_dim if inferred_dim is not None else fallback_dim
                features.append(torch.zeros(dim, dtype=torch.float32))
                continue

            emb = np.asarray(face.embedding, dtype=np.float32)
            if emb.ndim != 1:
                emb = emb.reshape(-1)
            inferred_dim = emb.shape[0]
            features.append(torch.from_numpy(emb))

        if not features:
            continue

        scene_tensor = torch.stack(features, dim=0)
        with open(save_path, "wb") as f:
            pickle.dump(scene_tensor, f)
        saved_files += 1

    print(f"Done. saved_scenes={saved_files}, total_images={total_images}, no_face_or_bad={miss_faces}")
    print(f"Output dir: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract face embeddings with InsightFace.")
    parser.add_argument("--input-dir", default="data/face_images", help="Input folder containing scene subfolders.")
    parser.add_argument("--output-base", default="data", help="Parent folder for output features.")
    parser.add_argument("--model-name", default="antelopev2", help="InsightFace model name.")
    parser.add_argument("--model-root", default="pretrained", help="Folder to cache/download model weights.")
    parser.add_argument("--det-size", type=int, default=640, help="Face detector input size.")
    parser.add_argument("--fallback-dim", type=int, default=512, help="Embedding dim used when face is missing.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scene feature files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.join(args.output_base, args.model_name)
    use_cuda = torch.cuda.is_available() and (not args.cpu)

    os.makedirs(args.model_root, exist_ok=True)
    app = _build_app(
        model_name=args.model_name,
        model_root=args.model_root,
        det_size=args.det_size,
        use_cuda=use_cuda,
    )
    extract(
        input_dir=args.input_dir,
        output_dir=output_dir,
        app=app,
        fallback_dim=args.fallback_dim,
        overwrite=args.overwrite,
    )
