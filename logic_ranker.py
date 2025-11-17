# anomaly_logic_ranker.py
import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import json  # <-- add this near the top with other imports

# If you have the global list in backend, you can import it; otherwise define a default:
try:
    from backend import ANOMALY_POSITIONS  # for canonical ordering / available labels
    DEFAULT_LABEL_ORDER = list(ANOMALY_POSITIONS.keys())
except Exception:
    DEFAULT_LABEL_ORDER = [
        "Dead Body", "Door Anomaly", "Extra Object", "Image Anomaly", "Intruder",
        "Missing Object", "Object Manipulation", "Object Movement", "Object Replacement",
        "Other",
    ]
# COMPARISON
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")
CLASSIFICATION_MAP = os.path.join(BASE_DIR, "classification_map.json")


def _dhash_signature(image_path: str) -> Optional[str]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return f"{val:016x}"


def _hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _load_map() -> Dict[str, str]:
    if not os.path.exists(CLASSIFICATION_MAP):
        return {}
    try:
        with open(CLASSIFICATION_MAP, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# -----------------------------
# Utilities
# -----------------------------
def _component_stats(mask: np.ndarray):
    """Return connected components info and some convenient aggregates."""
    if mask is None or mask.size == 0:
        return dict(count=0, area_frac=0.0, largest_area_frac=0.0,
                    bbox_frac=0.0, aspect_ratio=0.0, centroid=(None, None))

    H, W = mask.shape[:2]
    area_img = float(H * W)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return dict(count=0, area_frac=0.0, largest_area_frac=0.0,
                    bbox_frac=0.0, aspect_ratio=0.0, centroid=(None, None))

    # drop background (0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    xs, ys, ws, hs = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3]
    bbox_areas = (ws * hs).astype(np.float32)

    total_white = float((mask > 0).sum())
    area_frac = total_white / area_img

    i_max = int(np.argmax(areas))
    largest_area = float(areas[i_max])
    largest_area_frac = largest_area / area_img

    # Largest component bbox stats
    wL, hL = float(ws[i_max]), float(hs[i_max])
    bbox_frac = (wL * hL) / area_img if area_img > 0 else 0.0
    aspect_ratio = (hL / wL) if wL > 1 else 0.0

    cx, cy = cents[i_max]
    centroid = (float(cx) / W, float(cy) / H)  # normalized [0..1]

    return dict(
        count=int(len(areas)),
        area_frac=float(area_frac),
        largest_area_frac=float(largest_area_frac),
        bbox_frac=float(bbox_frac),
        aspect_ratio=float(aspect_ratio),
        centroid=centroid,
    )

def _signed_intensity_shift(baseline_bgr: Optional[np.ndarray],
                            current_bgr:  Optional[np.ndarray]) -> float:
    """
    Mean signed change (curr - base) in grayscale. Positive => got brighter (often 'Extra Object'),
    Negative => got darker/missing (often 'Missing Object'). 0 when not available.
    """
    if baseline_bgr is None or current_bgr is None:
        return 0.0
    base_g = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(current_bgr,  cv2.COLOR_BGR2GRAY)
    if base_g.shape != curr_g.shape:
        target = (curr_g.shape[1], curr_g.shape[0])
        base_g = cv2.resize(base_g, target, interpolation=cv2.INTER_AREA)
    return float(curr_g.astype(np.int16).mean() - base_g.astype(np.int16).mean())

def _edge_delta(baseline_bgr: Optional[np.ndarray],
                current_bgr:  Optional[np.ndarray]) -> float:
    """How much edge energy changed (Canny). Larger can imply 'Image Anomaly' or 'Replacement'."""
    if baseline_bgr is None or current_bgr is None:
        return 0.0
    base_g = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(current_bgr,  cv2.COLOR_BGR2GRAY)
    if base_g.shape != curr_g.shape:
        target = (curr_g.shape[1], curr_g.shape[0])
        base_g = cv2.resize(base_g, target, interpolation=cv2.INTER_AREA)
    e0 = cv2.Canny(base_g, 50, 120).mean()
    e1 = cv2.Canny(curr_g, 50, 120).mean()
    return float(abs(e1 - e0))

def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    vals = np.array([max(0.0, v) for v in scores.values()], dtype=np.float32)
    s = float(vals.sum())
    if s <= 1e-8:
        # fallback: uniform small probability
        n = max(1, len(vals))
        return {k: 1.0 / n for k in scores.keys()}
    return {k: float(v) / s for k, v in scores.items()}

# -----------------------------
# Heuristic label scoring
# -----------------------------
def _score_labels(mask_stats: dict,
                  signed_shift: float,
                  edge_change: float,
                  room: Optional[str],
                  labels: List[str]) -> Dict[str, float]:
    """
    Produce a raw score per label using simple, explainable heuristics.
    You can tune these safely at runtime.
    """
    area = mask_stats["area_frac"]           # fraction of region marked as change
    largest = mask_stats["largest_area_frac"]
    bboxf = mask_stats["bbox_frac"]
    ar    = mask_stats["aspect_ratio"]       # H/W of largest component
    count = mask_stats["count"]
    cx, cy = mask_stats["centroid"]

    scores = {lab: 0.0 for lab in labels}

    # --- Intruder ---
    # One tall-ish blob, moderate area, often around mid frame
    if "Intruder" in scores:
        s = 0.0
        if 0.012 <= largest <= 0.25:
            s += 1.0
        if ar >= 1.6:
            s += 0.8
        if count <= 3:
            s += 0.4
        scores["Intruder"] = s

    # --- Extra Object ---
    # New compact bright blob, small-to-moderate area, positive brightening
    if "Extra Object" in scores:
        s = 0.0
        if 0.003 <= largest <= 0.05:
            s += 0.9
        if signed_shift > 2.0:   # brighter
            s += 0.7
        if count <= 6:
            s += 0.2
        scores["Extra Object"] = max(s, 0.0)

    # --- Missing Object ---
    # Region darker overall (negative shift) and/or a hole-like change
    if "Missing Object" in scores:
        s = 0.0
        if signed_shift < -2.0:
            s += 1.0
        if area < 0.12 and count <= 4:
            s += 0.3
        scores["Missing Object"] = max(s, 0.0)

    # --- Object Movement ---
    # Similar energy but position shift → compact area, edges similar
    if "Object Movement" in scores:
        s = 0.0
        if 0.002 <= area <= 0.08:
            s += 0.6
        if edge_change < 6.0:
            s += 0.4
        scores["Object Movement"] = s

    # --- Object Manipulation ---
    # Slight visual tweak of an object → small area, small shift
    if "Object Manipulation" in scores:
        s = 0.0
        if 0.001 <= area <= 0.05:
            s += 0.6
        if abs(signed_shift) <= 4.0:
            s += 0.3
        scores["Object Manipulation"] = s

    # --- Object Replacement ---
    # Similar area but different texture/edge energy; often medium area
    if "Object Replacement" in scores:
        s = 0.0
        if 0.006 <= area <= 0.15:
            s += 0.6
        if edge_change >= 6.0:
            s += 0.6
        scores["Object Replacement"] = s

    # --- Image Anomaly ---
    # Global/large-area change (color tone, posterization, etc.)
    if "Image Anomaly" in scores:
        s = 0.0
        if area >= 0.15 or count >= 20:
            s += 1.2
        if edge_change >= 8.0:
            s += 0.4
        scores["Image Anomaly"] = s

    # --- Door Anomaly ---
    # Often tall rectangular swath near door region; boost if tall aspect
    if "Door Anomaly" in scores:
        s = 0.0
        if ar >= 2.0 and bboxf >= 0.01:
            s += 0.7
        if 0.003 <= area <= 0.12:
            s += 0.4
        scores["Door Anomaly"] = s

    # --- Dead Body ---
    # Medium-ish blob lying, not extremely tall; moderate area
    if "Dead Body" in scores:
        s = 0.0
        if 0.01 <= largest <= 0.12:
            s += 0.8
        if 0.8 <= ar <= 1.8:
            s += 0.3
        scores["Dead Body"] = s

    # --- Other ---
    if "Other" in scores:
        s = 0.1 + 0.3 * float(area > 0.0)
        scores["Other"] = s

    return scores

# -----------------------------
# Public entry point
# -----------------------------
def rank_anomaly_types(
    heatmap_path: str,
    *,
    bw_mask: Optional[np.ndarray] = None,
    baseline_bgr: Optional[np.ndarray] = None,
    current_bgr:  Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    room: Optional[str] = None,
    density_cap: int = 64,           # kept for selector API compatibility
    known_conf_threshold: float = 0.60,
    max_sig_distance: int = 6,       # <-- JSON match threshold (like before)
) -> Tuple[bool, List[Dict[str, object]]]:
    """
    Return (is_known, ranked_details)
      ranked_details = [{"label": str, "distance": Optional[int], "confidence": float}, ...]
    Priority:
      1) If dhash matches classification_map.json within max_sig_distance -> return that label.
      2) Else compute heuristic scores from mask / baseline/current features.
    """

    # 0) labels list
    if labels is None:
        labels = DEFAULT_LABEL_ORDER

    # 1) --- MEMORY LOOKUP (classification_map.json) ---
    # Try to match the current heatmap by dhash against previously confirmed signatures.
    mapping = _load_map()
    sig = _dhash_signature(heatmap_path)
    if sig and mapping:
        best_label, best_dist = None, 1_000_000
        for known_sig, label in mapping.items():
            d = _hamming(sig, known_sig)
            if d < best_dist:
                best_label, best_dist = label, d

        if best_label is not None and best_dist <= max_sig_distance:
            # Winner first, then the rest
            ranked = [{"label": best_label, "distance": best_dist, "confidence": 1.0}]
            ranked += [
                {"label": L, "distance": None, "confidence": 0.0}
                for L in labels if L != best_label
            ]
            return True, ranked

    # 2) --- HEURISTICS PATH (no confident JSON match) ---
    # Build/derive a binary mask (white=change) if not provided
    mask = None
    if bw_mask is not None:
        mask = (bw_mask > 0).astype(np.uint8) * 255
    else:
        img = cv2.imread(heatmap_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t_otsu, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            t = max(t_otsu, 200)  # keep only bright whites from your overlay
            _, mask = cv2.threshold(gray, int(t), 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)

    if mask is None:
        # Nothing to go on → uniform small probabilities
        ranked = [{"label": L, "distance": None, "confidence": 1.0 / len(labels)} for L in labels]
        return False, ranked

    # 3) Feature extraction
    ms = _component_stats(mask)
    signed_shift = _signed_intensity_shift(baseline_bgr, current_bgr)
    edge_change  = _edge_delta(baseline_bgr, current_bgr)

    # 4) Score & normalize
    raw_scores = _score_labels(ms, signed_shift, edge_change, room, labels)
    confs = _normalize_scores(raw_scores)

    ranked = [{"label": L, "distance": None, "confidence": float(confs.get(L, 0.0))} for L in labels]
    ranked.sort(key=lambda x: x["confidence"], reverse=True)

    is_known = ranked[0]["confidence"] >= known_conf_threshold
    return is_known, ranked