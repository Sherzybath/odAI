# selector.py
import os
import json
import cv2
import pyautogui
from typing import Tuple, Optional, Dict
import numpy as np
from backend import move_cursor_to_anomaly, move_cursor_to_dropdown_top_any_side, ANOMALY_POSITIONS
import time

try:
    from backend import read_center_status as _backend_read_center_status
except Exception:
    _backend_read_center_status = None

BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")
CLASSIFICATION_MAP = os.path.join(BASE_DIR, "classification_map.json")
DEFAULT_LABEL_ORDER = list(ANOMALY_POSITIONS.keys())


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


def _interpret_center_status(msg: str) -> str:
    """
    Normalize the center-screen status message into one of:
    - 'detected'  -> e.g., 'anomaly detected in the center of the screen'
    - 'none'      -> e.g., 'no anomalies found in the center of the screen'
    - 'unknown'   -> anything else / empty
    """
    if not msg:
        return 'unknown'
    m = msg.strip().lower()
    if "anomaly detected" in m and "center" in m:
        return 'detected'
    if "no anomalies" in m and "center" in m:
        return 'none'
    return 'unknown'

def select_and_click_until_detected(
    ranked_types,
    move_cursor_to_anomaly,
    read_center_status,
    click_fn=None,
    wait_seconds: int = 5,
):
    """
    Iterate over ranked types (most likely -> least likely).
    For each:
      1) move cursor to that anomaly candidate
      2) click
      3) wait `wait_seconds`
      4) read center status
         - if status says 'no anomalies...' -> try next type
         - if status says 'anomaly detected...' -> stop
    Prints each type with confidence and the outcome.
    """
    # Lazy import to avoid hard dependency if caller injects a custom click_fn
    if click_fn is None:
        try:
            import pyautogui
            click_fn = lambda: pyautogui.click()
        except Exception:
            click_fn = lambda: None

    print("=== Candidate types (most likely → least likely) ===")
    for idx, (t, conf) in enumerate(ranked_types, start=1):
        print(f"{idx:02d}. {t}: {conf:.3f}")

    for idx, (t, conf) in enumerate(ranked_types, start=1):
        print(f"\n[TRY {idx}] Moving to '{t}' (conf={conf:.3f}) …")
        try:
            move_cursor_to_anomaly(t)
        except Exception as e:
            print(f"[TRY {idx}] move_cursor_to_anomaly('{t}') failed: {e}")
            continue

        try:
            click_fn()
            print(f"[TRY {idx}] Clicked. Waiting {wait_seconds}s …")
        except Exception as e:
            print(f"[TRY {idx}] click failed: {e}")
        time.sleep(wait_seconds)

        try:
            status_raw = read_center_status() or ""
        except Exception as e:
            print(f"[TRY {idx}] read_center_status failed: {e}")
            status_raw = ""

        norm = _interpret_center_status(status_raw)
        print(f"[TRY {idx}] Center status: '{status_raw}' → {norm}")

        if norm == 'detected':
            print(f"[SUCCESS] Anomaly confirmed after selecting '{t}'.")
            return
        elif norm == 'none':
            print(f"[TRY {idx}] No anomaly with '{t}'. Trying next …")
            continue
        else:
            print(f"[TRY {idx}] Unknown/ambiguous status. Proceeding to next candidate …")
            continue

    print("[END] Exhausted all candidates without a confirmed anomaly.")


def _rank_labels_for_heatmap(heatmap_path: str,
                             max_distance: int = 6,
                             distance_cap: int = 64):
    """
    Returns:
      is_known (bool),
      ranked_details: list of dicts sorted best→worst like:
        [{"label": str, "distance": int or None, "confidence": float in [0,1]}]
    """
    sig = _dhash_signature(heatmap_path)
    if sig is None:
        # No signature -> fall back to default order with zero confidence
        ranked_details = [{"label": L, "distance": None, "confidence": 0.0}
                          for L in DEFAULT_LABEL_ORDER]
        return False, ranked_details

    mapping = _load_map()
    if not mapping:
        ranked_details = [{"label": L, "distance": None, "confidence": 0.0}
                          for L in DEFAULT_LABEL_ORDER]
        return False, ranked_details

    # Best (smallest) Hamming distance per label
    best_per_label = {}
    for known_sig, label in mapping.items():
        d = _hamming(sig, known_sig)
        cur = best_per_label.get(label)
        if cur is None or d < cur:
            best_per_label[label] = d

    # Build details for seen labels (with distance) and unseen (None distance)
    seen = []
    for L, d in best_per_label.items():
        # confidence: smaller distance → closer to 1; cap by distance_cap
        d_cap = min(d, distance_cap)
        conf = max(0.0, 1.0 - (d_cap / float(distance_cap)))
        seen.append({"label": L, "distance": d, "confidence": conf})

    unseen = [{"label": L, "distance": None, "confidence": 0.0}
              for L in DEFAULT_LABEL_ORDER if L not in best_per_label]

    # Sort seen by distance asc (i.e., confidence desc), then append unseen
    seen.sort(key=lambda x: x["distance"])
    ranked_details = seen + unseen

    # Decide "known" by threshold on the best seen distance
    is_known = bool(seen) and (seen[0]["distance"] <= max_distance)
    return is_known, ranked_details
def select_by_rank(coords,
                   heatmap_path: str,
                   logger=None,
                   max_distance: int = 6,
                   distance_cap: int = 64,
                   hold_seconds: float = 2.0,
                   step_px: int = 60,           # kept for compatibility; unused in click-loop
                   pause: float = 0.18,          # kept for compatibility; unused in click-loop
                   read_center_status=None,
                   click_fn=None,
                   wait_seconds: int = 5) -> None:
    """
    1) Rank labels for the given heatmap (with confidences).
    2) Log all labels with confidence.
    3) For each label (most likely → least likely):
         - move to that anomaly type in the dropdown,
         - click,
         - wait `wait_seconds`,
         - read center status; stop on 'anomaly detected', otherwise continue.
    """

    def log(msg: str):
        if callable(logger):
            logger(msg)
        else:
            print(msg)

    # Robust lazy auto-wire for backend.read_center_status
    if read_center_status is None:
        try:
            import backend as _be
            read_center_status = getattr(_be, "read_center_status", None)
            if read_center_status is None:
                import importlib
                _be = importlib.reload(_be)  # pick up newly added function
                read_center_status = getattr(_be, "read_center_status", None)
        except Exception:
            read_center_status = None

    if read_center_status is None or not callable(read_center_status):
        log("read_center_status callable is required for detection loop.")
        return

    is_known, ranked_details = _rank_labels_for_heatmap(
        heatmap_path, max_distance=max_distance, distance_cap=distance_cap
    )

    if not ranked_details:
        log("No labels found to select.")
        return

    # Log the full ranking with confidences
    log("— Anomaly type ranking —")
    for item in ranked_details:
        lab  = item["label"]
        d    = item["distance"]
        conf = item["confidence"]
        if d is None:
            log(f"  {lab:>20}  conf={conf:.2f}  (no prior)")
        else:
            log(f"  {lab:>20}  conf={conf:.2f}  dist={d}")
    log(f"Known match within threshold: {is_known}")

    # Prepare (label, confidence) tuples for the click-iterate loop
    ranked_types = [(item["label"], float(item["confidence"])) for item in ranked_details]

    # Wrapper to move to the specific label inside your dropdown flow
    def _move_to_label(label: str):
        move_cursor_to_anomaly(coords,
                               hold_seconds=hold_seconds,
                               dropdown=True,
                               anomaly_type=label,
                               logger=logger)

    # Run the click → wait → read loop until detection or exhaustion
    select_and_click_until_detected(
        ranked_types=ranked_types,
        move_cursor_to_anomaly=_move_to_label,
        read_center_status=read_center_status,
        click_fn=click_fn,
        wait_seconds=wait_seconds,
    )