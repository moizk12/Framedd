from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


# Path B (semantic) - CLIP zero shot
# not fine tuning, just score against a small tag list


@dataclass
class ClipResult:
    primary_scene: str
    confidence: float
    top_k: List[Dict[str, float]]


def get_default_clip_labels():
    # keep list small + class friendly (can add/remove)
    return [
        "urban architecture",
        "street scene",
        "indoor room",
        "mountain landscape",
        "forest greenery",
        "beach ocean",
        "portrait photo",
        "night city",
        "car road",
        "abstract art",
    ]


_clip_model = None
_clip_proc = None


def _load_clip(device: str):
    global _clip_model, _clip_proc
    if _clip_model is None or _clip_proc is None:
        # common CLIP backbone, not huge
        model_id = "openai/clip-vit-base-patch32"
        _clip_model = CLIPModel.from_pretrained(model_id)
        _clip_proc = CLIPProcessor.from_pretrained(model_id)
        _clip_model.eval()
        _clip_model.to(device)
    return _clip_model, _clip_proc


def classify_clip(image_path: str, labels: List[str] | None = None, topk: int = 3) -> Dict[str, Any]:
    """
    simple clip classify
    output dict (json safe)
    """
    if labels is None:
        labels = get_default_clip_labels()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc = _load_clip(device)

    img = Image.open(image_path).convert("RGB")
    inputs = proc(text=labels, images=img, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # logits per image: shape (1, n_labels)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=0)

    probs_list = probs.detach().cpu().numpy().tolist()

    scored = list(zip(labels, probs_list))
    scored.sort(key=lambda x: x[1], reverse=True)

    topk = max(1, min(int(topk), len(scored)))
    top = scored[:topk]

    primary_scene, conf = top[0][0], float(top[0][1])
    top_k = [{"label": lab, "confidence": float(p)} for lab, p in top]

    return {
        "primary_scene": primary_scene,
        "confidence": round(conf, 6),
        "top_k": top_k,
        "labels_used": labels,  # helps grading (shows its fixed list)
        "device": device,
    }

