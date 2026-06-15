"""
Class-based dimension priors for HTX-domain objects.

Pipeline:
    1. CLIP zero-shot classifier picks the most likely object class from a fixed list
    2. Each class has plausible (min_m, max_m, median_m) for its longest real-world dim
    3. sanity_check() compares the geometric prediction to the prior range:
        - within range → accept, no change
        - outside range but within 2x → flag confidence down, no override
        - way outside (>2x off) → snap to class median, mark "prior_snap"

This corrects the most common failure mode of view-aligned monocular scaling:
the depth network over- or under-estimating object distance leads to scale
errors of 2-4x for vehicles. The class prior bounds that error.

References:
    CLIP: Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision", ICML 2021. openai/clip-vit-base-patch32 (Apache 2.0).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Class priors table — longest_m dimensions per class
# ----------------------------------------------------------------------------
# Each entry: (min_m, max_m, median_m) for the LONGEST real-world dimension.
# Curated for HTX domain (police / SCDF / military / public infrastructure).
# Source: published manufacturer specs and standard class ranges.

CLASS_PRIORS: dict[str, dict] = {
    # ── Vehicles, civilian ────────────────────────────────────────
    "sedan car":               {"min_m": 3.8, "max_m": 5.5,  "median_m": 4.7},
    "suv or crossover vehicle":{"min_m": 4.2, "max_m": 5.5,  "median_m": 4.8},
    "pickup truck":            {"min_m": 5.0, "max_m": 6.5,  "median_m": 5.6},
    "van or minivan":          {"min_m": 4.8, "max_m": 6.5,  "median_m": 5.5},
    "delivery truck":          {"min_m": 6.0, "max_m": 12.0, "median_m": 8.5},
    "heavy truck or trailer":  {"min_m": 12.0,"max_m": 25.0, "median_m": 16.5},
    "motorcycle":              {"min_m": 1.8, "max_m": 2.5,  "median_m": 2.2},
    "bicycle":                 {"min_m": 1.5, "max_m": 2.0,  "median_m": 1.7},
    "scooter or moped":        {"min_m": 1.6, "max_m": 2.1,  "median_m": 1.9},
    "single decker bus":       {"min_m": 9.0, "max_m": 13.0, "median_m": 11.5},
    "double decker bus":       {"min_m": 10.0,"max_m": 13.0, "median_m": 12.0},

    # ── Vehicles, emergency / military ────────────────────────────
    "ambulance":               {"min_m": 5.5, "max_m": 8.0,  "median_m": 6.7},
    "fire engine":             {"min_m": 8.0, "max_m": 12.0, "median_m": 9.8},
    "police car":              {"min_m": 4.4, "max_m": 5.3,  "median_m": 4.85},
    "armored military vehicle":{"min_m": 5.5, "max_m": 8.5,  "median_m": 7.0},
    "tank":                    {"min_m": 7.0, "max_m": 10.0, "median_m": 8.5},
    "missile launcher vehicle":{"min_m": 6.0, "max_m": 12.0, "median_m": 7.5},
    "patrol boat":             {"min_m": 8.0, "max_m": 25.0, "median_m": 14.0},

    # ── People ───────────────────────────────────────────────────
    "person standing":         {"min_m": 1.5, "max_m": 2.0,  "median_m": 1.72},
    "person sitting":          {"min_m": 0.9, "max_m": 1.3,  "median_m": 1.10},

    # ── Equipment, weapons, gear ─────────────────────────────────
    "rifle or long gun":       {"min_m": 0.7, "max_m": 1.3,  "median_m": 1.00},
    "handgun or pistol":       {"min_m": 0.15,"max_m": 0.30, "median_m": 0.22},
    "backpack":                {"min_m": 0.4, "max_m": 0.7,  "median_m": 0.55},
    "helmet":                  {"min_m": 0.22,"max_m": 0.32, "median_m": 0.27},
    "drone or quadcopter":     {"min_m": 0.3, "max_m": 1.5,  "median_m": 0.7},
    "wheeled robot":           {"min_m": 0.6, "max_m": 1.8,  "median_m": 1.2},
    "bomb disposal robot":     {"min_m": 0.8, "max_m": 1.6,  "median_m": 1.30},
    "traffic cone":            {"min_m": 0.4, "max_m": 1.0,  "median_m": 0.71},
    "fire extinguisher":       {"min_m": 0.4, "max_m": 0.9,  "median_m": 0.55},
    "fire hydrant":            {"min_m": 0.5, "max_m": 1.0,  "median_m": 0.75},
    "bollard":                 {"min_m": 0.6, "max_m": 1.2,  "median_m": 0.9},
    "traffic light":           {"min_m": 0.8, "max_m": 1.5,  "median_m": 1.1},

    # ── Structures ───────────────────────────────────────────────
    "guard booth or kiosk":    {"min_m": 1.5, "max_m": 3.5,  "median_m": 2.3},
    "information kiosk":       {"min_m": 1.0, "max_m": 2.5,  "median_m": 1.8},
    "boom barrier gate":       {"min_m": 3.0, "max_m": 6.5,  "median_m": 4.5},
    "fence section":           {"min_m": 2.0, "max_m": 5.0,  "median_m": 3.0},
    "chair":                   {"min_m": 0.6, "max_m": 1.4,  "median_m": 0.9},
    "table or desk":           {"min_m": 0.8, "max_m": 2.2,  "median_m": 1.4},
    "container or box":        {"min_m": 0.3, "max_m": 1.5,  "median_m": 0.8},
    "shipping container":      {"min_m": 6.0, "max_m": 13.0, "median_m": 6.06},

    # ── Generic fallback (very wide bounds, won't trigger snap) ──
    "unknown object":          {"min_m": 0.3, "max_m": 25.0, "median_m": 2.0},
}


# ----------------------------------------------------------------------------
# CLIP zero-shot classifier (singleton, lazy-loaded)
# ----------------------------------------------------------------------------

class _ClipClassifier:
    _model = None
    _processor = None
    _model_name = "openai/clip-vit-base-patch32"
    _classes: list[str] = list(CLASS_PRIORS.keys())
    _device = "cuda"

    @classmethod
    def get(cls):
        if cls._model is None:
            logger.info("loading CLIP zero-shot classifier (%s)", cls._model_name)
            from transformers import CLIPProcessor, CLIPModel
            cls._processor = CLIPProcessor.from_pretrained(cls._model_name)
            cls._model = CLIPModel.from_pretrained(cls._model_name).to(cls._device).eval()
        return cls._model, cls._processor

    @classmethod
    def unload(cls):
        cls._model = None; cls._processor = None
        torch.cuda.empty_cache()


@dataclass
class ClassPrediction:
    label: str
    confidence: float            # softmax probability of top-1
    top3: list[tuple[str, float]] # diagnostic
    min_m: float
    max_m: float
    median_m: float


def classify_object(image_path: str) -> ClassPrediction:
    """Zero-shot classify the object in the image against the CLASS_PRIORS labels."""
    model, processor = _ClipClassifier.get()
    img = Image.open(image_path).convert("RGB")
    classes = list(CLASS_PRIORS.keys())
    # CLIP works best with descriptive prompts
    prompts = [f"a photo of a {c}" for c in classes]
    with torch.no_grad():
        inputs = processor(text=prompts, images=img, return_tensors="pt", padding=True).to(_ClipClassifier._device)
        outputs = model(**inputs)
        # logits_per_image: (1, n_classes)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    top1 = classes[order[0]]
    top3 = [(classes[i], float(probs[i])) for i in order[:3]]
    prior = CLASS_PRIORS[top1]
    return ClassPrediction(
        label=top1, confidence=float(probs[order[0]]), top3=top3,
        min_m=prior["min_m"], max_m=prior["max_m"], median_m=prior["median_m"],
    )


# ----------------------------------------------------------------------------
# Sanity check: clip prediction to class plausible range or snap if way off
# ----------------------------------------------------------------------------

@dataclass
class PriorCheck:
    in_range: bool                    # was the geometric prediction inside the class plausible range
    snap_applied: bool                # did we actually blend toward the class prior
    pre_snap_longest_m: float         # original geometric prediction
    post_snap_longest_m: float        # blended prediction
    snap_scale_factor: float          # multiplier to apply to existing GLB to reach post_snap
    confidence_modifier: str          # 'unchanged' | 'downgrade' | 'downgrade2'
    blend_alpha: float                # weight given to the class prior (0=trust prediction, 1=trust prior)


def sanity_check(predicted_longest_m: float,
                 class_pred: ClassPrediction,
                 blend_strength: float = 1.0,
                 min_classifier_conf: float = 0.20) -> PriorCheck:
    """Bayesian-style blend of geometric prediction with class prior.

    Logic:
      - if prediction is inside [min_m, max_m]: do nothing (in_range=True)
      - else target = nearest range bound (not median — more conservative)
      - blend: final = (1 - α) * predicted + α * target
        α = classifier_confidence × blend_strength (clamped to [0, 1])
      - if classifier_confidence < min_classifier_conf: skip the blend, just flag
        (we don't want to override the prediction based on an uncertain classifier)

    Higher classifier confidence → stronger correction toward the class range.
    Low confidence → leave the prediction alone (just flag confidence down).
    """
    p = predicted_longest_m
    lo, hi, med = class_pred.min_m, class_pred.max_m, class_pred.median_m
    if lo <= p <= hi:
        return PriorCheck(True, False, p, p, 1.0, "unchanged", 0.0)

    # Uncertain classifier — don't override, just flag down
    if class_pred.confidence < min_classifier_conf:
        return PriorCheck(False, False, p, p, 1.0, "downgrade", 0.0)

    # Snap toward the nearest bound (conservative — bigger correction if more out of range)
    target = lo if p < lo else hi
    alpha = float(np.clip(class_pred.confidence * blend_strength, 0.0, 1.0))
    final = (1.0 - alpha) * p + alpha * target
    snap_factor = final / max(p, 1e-6)

    # Heavy mismatch → also pull a fraction toward median (downgrade2)
    off_factor = (lo / p) if p < lo else (p / hi)
    if off_factor > 2.0 and class_pred.confidence > 0.4:
        # blend an extra step toward the median for confident hard misses
        final = 0.5 * final + 0.5 * med
        snap_factor = final / max(p, 1e-6)
        return PriorCheck(False, True, p, final, snap_factor, "downgrade2", alpha)

    return PriorCheck(False, True, p, final, snap_factor, "downgrade", alpha)


# CLI for quick testing
if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO)
    img = sys.argv[1]
    pred = classify_object(img)
    print(json.dumps({
        "label": pred.label,
        "confidence": pred.confidence,
        "top3": pred.top3,
        "plausible_range_m": [pred.min_m, pred.max_m],
        "median_m": pred.median_m,
    }, indent=2))
