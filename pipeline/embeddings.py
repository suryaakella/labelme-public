import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_tokenizer = None


def _get_clip():
    global _clip_model, _clip_preprocess, _tokenizer
    if _clip_model is None:
        import open_clip
        from config import settings

        logger.info(f"Loading CLIP model: {settings.clip_model} ({settings.clip_pretrained})")
        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(settings.clip_model)
        model.eval()

        _clip_model = model
        _clip_preprocess = preprocess
        _tokenizer = tokenizer

    return _clip_model, _clip_preprocess, _tokenizer


class CLIPEmbedder:
    def embed_image(self, image_path: str) -> np.ndarray:
        model, preprocess, _ = _get_clip()
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            features = model.encode_image(image)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()

    def embed_text(self, text: str) -> np.ndarray:
        model, _, tokenizer = _get_clip()
        tokens = tokenizer([text])

        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()

    def embed_video(self, frame_paths: List[str]) -> np.ndarray:
        if not frame_paths:
            raise ValueError("No frames provided for video embedding")

        model, preprocess, _ = _get_clip()
        images = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in frame_paths
        ])

        with torch.no_grad():
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        # Average pool across frames and re-normalize
        avg = features.mean(dim=0)
        avg = avg / avg.norm()
        return avg.cpu().numpy()

    def embed_images_batch(self, image_paths: List[str]) -> List[np.ndarray]:
        if not image_paths:
            return []

        model, preprocess, _ = _get_clip()
        images = torch.stack([
            preprocess(Image.open(p).convert("RGB")) for p in image_paths
        ])

        with torch.no_grad():
            features = model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        return [features[i].cpu().numpy() for i in range(len(image_paths))]
