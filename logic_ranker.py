import os, json, cv2, numpy as np
from typing import Dict, List, Tuple, Optional

# ---- anomaly label universe (keep in sync with your dropdown) ----
ANOMALY_LABELS = [
    "Dead Body",
    "Door Anomaly",
    "Extra Object",
    "Image Anomaly",
    "Intruder",
    "Missing Object",
    "Object Manipulation",
    "Object Movement",
    "Object Replacement",
    "Other",
]

# ---- priors by region name (class_name in your template regions) ----
# feel free to add more keys from your dataset: "painting", "shelf", "table", "door", "wall", "floor", ...
REGION_PRIORS: Dict[str, Dict[str, float]] = {
    "frame":            {"Image Anomaly": 0.65, "Object Replacement": 0.15, "Other": 0.10, "Object Movement": 0.10},
    "painting":         {"Image Anomaly": 0.60, "Object Replacement": 0.20, "Other": 0.10, "Object Movement": 0.10},
    "door":             {"Door Anomaly": 0.65, "Object Movement": 0.20, "Other": 0.15},
    "table":            {"Extra Object": 0.40, "Missing Object": 0.25, "Object Manipulation": 0.20, "Object Movement": 0.15},
    "shelf":            {"Extra Object": 0.45, "Missing Object": 0.30, "Object Manipulation": 0.15, "Other": 0.10},
    "floor":            {"Intruder": 0.30, "Extra Object": 0.30, "Object Movement": 0.25, "Other": 0.15},
    "wall":             {"Image Anomaly": 0.35, "Other": 0.25, "Object Replacement": 0.20, "Object Movement": 0.20},
}

# default priors if region unknown
DEFAULT_REGION_PRIOR = {"Other": 0.34, "Extra Object": 0.22, "Object Movement": 0.22, "Image Anomaly": 0.22}

# ---- weights for each signal (sum roughly ~1; tune freely) ----
WEIGHTS = {
    "signature": 0.35,   # dhash distance → similarity
    "region":    0.20,   # region prior
    "shape":     0.20,   # area/AR/solidity
    "temporal":  0.15,   # persistence vs spike
    "colorish":  0.10,   # uniformity / warm hues bias
}

# ---- utilities ----------------------------------------------------

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

def _load_map(classification_map_path: str) -> Dict[str, str]:
    if not os.path.exists(classification_map_path):
        return {}
    try:
        with open(classification_map_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _signature_scores(heatmap_path: str, classification_map_path: str, distance_cap: int = 64) -> Dict[str, float]:
    """return per-label similarity score in [0,1] derived from dhash nearest distances"""
    sig = _dhash_signature(heatmap_path)
    if sig is None:
        return {lbl: 0.0 for lbl in ANOMALY_LABELS}

    mapping = _load_map(classification_map_path)
    if not mapping:
        return {lbl: 0.0 for lbl in ANOMALY_LABELS}

    # collect min distance per label
    best_per_label: Dict[str, int] = {}
    for known_sig, label in mapping.items():
        d = _hamming(sig, known_sig)
        cur = best_per_label.get(label)
        if cur is None or d < cur:
            best_per_label[label] = d

    scores = {lbl: 0.0 for lbl in ANOMALY_LABELS}
    for lbl, d in best_per_label.items():
        d_cap = min(d, distance_cap)
        scores[lbl] = max(0.0, 1.0 - (d_cap / float(distance_cap)))
    return scores

def _region_prior(region_name: str) -> Dict[str, float]:
    if not region_name:
        return DEFAULT_REGION_PRIOR
    r = region_name.lower()
    # choose best matching prior by substring
    for key in REGION_PRIORS:
        if key in r:
            return REGION_PRIORS[key]
    return DEFAULT_REGION_PRIOR

def _mask_features(bw: np.ndarray) -> Dict[str, float]:
    """extract simple shape/area features from mask"""
    H, W = bw.shape[:2]
    area_frac = float(cv2.countNonZero(bw)) / float(H * W + 1e-6)

    # largest component properties
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return {"area": area_frac, "ar": 1.0, "sol": 0.0}

    idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    x, y, w, h, a = stats[idx]
    ar = (h + 1e-6) / (w + 1e-6) if w > 0 else 1.0  # tall vs wide
    # solidity: area / convexArea approx via hull on contour
    comp = np.zeros_like(bw)
    comp[labels == idx] = 255
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sol = 0.0
    if contours:
        hull = cv2.convexHull(contours[0])
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            sol = float(a) / float(hull_area)
    return {"area": area_frac, "ar": float(ar), "sol": float(sol)}

def _shape_scores(feat: Dict[str, float]) -> Dict[str, float]:
    """
    heuristic mapping:
      - Intruder: larger area, tall-ish blob, decent solidity
      - Image Anomaly: moderate-to-large area, fairly solid, often rectangular-ish (solidity high)
      - Extra Object: small-to-moderate area, compact (solidity high)
      - Object Movement: very small area OR thin edges → low solidity
      - Missing Object: can be tricky; if area moderate but irregular, give light credit
    """
    area = feat["area"]       # 0..1
    ar   = feat["ar"]         # tall if > 1, wide if < 1
    sol  = feat["sol"]        # 0..1

    s = {lbl: 0.0 for lbl in ANOMALY_LABELS}

    # Intruder
    s["Intruder"] = np.clip((area - 0.008) * 40.0, 0, 1) * np.clip((ar - 0.8), 0, 1) * np.clip((sol - 0.4) * 2.0, 0, 1)

    # Image Anomaly
    s["Image Anomaly"] = np.clip((area - 0.004) * 35.0, 0, 1) * np.clip((sol - 0.5) * 2.0, 0, 1)

    # Extra Object
    s["Extra Object"] = np.clip(0.012 - area, 0, 0.012) / 0.012 * np.clip((sol - 0.4) * 2.0, 0, 1)

    # Object Movement (thin/edgey/small)
    s["Object Movement"] = np.clip(0.006 - area, 0, 0.006) / 0.006 * np.clip(0.6 - sol, 0, 0.6) / 0.6

    # Missing Object (weak heuristic)
    s["Missing Object"] = np.clip((0.02 - area), 0, 0.02) / 0.02 * np.clip((0.7 - sol), 0, 0.7) / 0.7 * 0.6

    # door anomaly – penalize unless region prior pushes it
    s["Door Anomaly"] = 0.1 * np.clip((area - 0.003) * 50.0, 0, 1)

    # object manipulation / replacement – light signals
    s["Object Manipulation"] = 0.25 * np.clip((area - 0.003) * 40.0, 0, 1)
    s["Object Replacement"]  = 0.25 * np.clip((area - 0.003) * 40.0, 0, 1)

    # other
    s["Other"] = 0.1
    return s

def _temporal_scores(history_flags: List[bool]) -> Dict[str, float]:
    """
    history_flags: e.g., last N polls where this region was 'hot' (True) vs 'cold' (False)
    persistent (many Trues) → Image/Missing/Manipulation
    spiky (few Trues) → Movement/Extra
    """
    if not history_flags:
        persist = 0.0
    else:
        persist = sum(history_flags) / float(len(history_flags))
    s = {lbl: 0.0 for lbl in ANOMALY_LABELS}
    s["Image Anomaly"] = persist
    s["Missing Object"] = persist * 0.8
    s["Object Manipulation"] = persist * 0.7
    s["Object Movement"] = (1.0 - persist) * 0.8
    s["Extra Object"] = (1.0 - persist) * 0.5
    s["Intruder"] = persist * 0.3  # intruder can persist but often appears suddenly
    s["Other"] = 0.2
    return s

def _colorish_scores(current_bgr: Optional[np.ndarray], baseline_bgr: Optional[np.ndarray]) -> Dict[str, float]:
    """
    quick uniformity measure: std of absdiff; lower std + higher mean → uniform patch (Image Anomaly)
    """
    s = {lbl: 0.0 for lbl in ANOMALY_LABELS}
    if current_bgr is None or baseline_bgr is None:
        return s
    base_g = cv2.cvtColor(baseline_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(current_bgr,  cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(base_g, curr_g).astype(np.float32)
    mean = float(diff.mean())
    std  = float(diff.std()) + 1e-6
    uniformity = np.clip(mean / (std * 6.0), 0.0, 1.0)  # bigger when mean>>std
    s["Image Anomaly"] = uniformity
    s["Object Movement"] = 1.0 - uniformity
    return s

# ---- public API ---------------------------------------------------

def rank_labels_for_anomaly(
    *,
    heatmap_path: str,
    classification_map_path: str,
    region_name: str,                         # from your template region["class_name"]
    bw_mask: np.ndarray,                      # from convert_to_black_white_fullsize
    history_hot_flags: List[bool] | None,     # last N booleans for this region (optional)
    baseline_bgr: Optional[np.ndarray] = None,
    current_bgr: Optional[np.ndarray] = None,
    distance_cap: int = 64,
    weights: Dict[str, float] = WEIGHTS,
    logger = print,
) -> List[Tuple[str, float]]:
    """
    Returns: list of (label, score) sorted DESC
    Also logs a short breakdown to help you tune.
    """
    # S1: signature
    sig_scores = _signature_scores(heatmap_path, classification_map_path, distance_cap)

    # S2: region prior
    prior = _region_prior(region_name)

    # S3: shape features → shape scores
    feat = _mask_features(bw_mask)
    shape_scores = _shape_scores(feat)

    # S4: temporal
    temporal_scores = _temporal_scores(history_hot_flags or [])

    # S5: colorish
    colorish = _colorish_scores(current_bgr, baseline_bgr)

    # combine with weights
    combined: Dict[str, float] = {}
    for lbl in ANOMALY_LABELS:
        combined[lbl] = (
            weights["signature"] * sig_scores.get(lbl, 0.0) +
            weights["region"]    * prior.get(lbl, 0.0) +
            weights["shape"]     * shape_scores.get(lbl, 0.0) +
            weights["temporal"]  * temporal_scores.get(lbl, 0.0) +
            weights["colorish"]  * colorish.get(lbl, 0.0)
        )

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    # log a compact breakdown
    if logger:
        logger("— ranking breakdown —")
        logger(f"region='{region_name}' feat={feat}")
        top_show = 6
        for lbl, sc in ranked[:top_show]:
            logger(f"  {lbl:>20}  score={sc:5.3f} | sig={sig_scores.get(lbl,0):.2f} rg={prior.get(lbl,0):.2f} sh={shape_scores.get(lbl,0):.2f} tm={temporal_scores.get(lbl,0):.2f} col={colorish.get(lbl,0):.2f}")

    return ranked
