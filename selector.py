# selector.py
import os
import json
import cv2
import pyautogui
from typing import Tuple, Optional, Dict
import numpy as np
from backend import (
    open_dropdown_and_anchor,          # NEW
    move_cursor_to_label_at_anchor,    # NEW
    reopen_dropdown_and_hover,         # NEW
    classify_heatmap,
    save_classification_signature,
    ANOMALY_POSITIONS,
)
import time 
from logic_ranker import rank_labels_for_anomaly

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
    Robustly normalize center-screen status:
      - 'detected' if message indicates an anomaly in progress/confirmed
      - 'none'     if message indicates no anomalies
      - 'unknown'  otherwise

    Tolerant to OCR noise, missing words like 'center', and phrasing variants.
    """
    if not msg:
        return 'unknown'

    # normalize
    m = msg.lower()
    # strip punctuation-like chars that OCR often injects
    keep = []
    for ch in m:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
    s = " ".join("".join(keep).split())  # collapse whitespace

    # convenience stems
    has_anomal  = ("anomal" in s)               # anomaly/anomalies
    has_detect  = ("detect" in s)               # detected/detecting/detection
    has_no      = (" no " in f" {s} ") or s.startswith("no ")
    has_found   = ("found" in s)
    has_center  = ("center" in s) or ("centre" in s)
    has_standby = ("stand by" in s) or ("standby" in s)

    # --- Positive / "detected" cases ---
    # 1) explicit anomaly + detected
    if has_anomal and has_detect:
        return 'detected'
    # 2) many UIs show "please stand by" during detection
    if has_anomal and has_standby:
        return 'detected'
    # 3) sometimes just "detected" (keep as a weaker positive)
    if has_detect and not has_no:
        return 'detected'

    # --- Negative / "none" cases ---
    # Allow variants like "no anomaly detected", "no anomalies found",
    # with or without "in the center of the screen".
    if has_no and has_anomal and (has_found or has_detect or has_center):
        return 'none'
    # Fallback: strong "no anomalies" even without other words
    if has_no and has_anomal:
        return 'none'

    return 'unknown'


def select_and_click_until_detected(
    ranked_types,
    move_cursor_to_anomaly,
    read_center_status,
    click_fn=None,
    wait_seconds: int = 5,
    poll_total: float = 2.5,   # extra time to poll after the initial wait
    poll_step: float = 0.25,   # poll interval
):
    """
    Try anomaly types (most likely → least likely), click each, and wait for detection.
    Returns the winning label on success, or None if none confirmed.
    """

    # Safe single-click helper (prevents stuck/down states)
    def _safe_click():
        try:
            import pyautogui
            pyautogui.mouseUp(button='left')
            pyautogui.click()
            pyautogui.mouseUp(button='left')
        except Exception:
            pass

    # Choose click fn
    if click_fn is None:
        click_fn = _safe_click

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

        # Click the candidate
        try:
            click_fn()
            print(f"[TRY {idx}] Clicked. Waiting {wait_seconds}s …")
        except Exception as e:
            print(f"[TRY {idx}] click failed: {e}")
        time.sleep(wait_seconds)

        # First read
        try:
            status_raw = read_center_status() or ""
        except Exception as e:
            print(f"[TRY {idx}] read_center_status failed: {e}")
            status_raw = ""

        norm = _interpret_center_status(status_raw)
        print(f"[TRY {idx}] Center status: '{status_raw}' → {norm}")

        # If unclear, poll a few times quickly (more robust than a single read)
        if norm == 'unknown' and poll_total > 0 and poll_step > 0:
            t_end = time.time() + poll_total
            while time.time() < t_end:
                time.sleep(poll_step)
                try:
                    sr = read_center_status() or ""
                except Exception as e:
                    print(f"[TRY {idx}] read_center_status (poll) failed: {e}")
                    sr = ""
                nn = _interpret_center_status(sr)
                print(f"[TRY {idx}] Poll status: '{sr}' → {nn}")
                if nn != 'unknown':
                    norm = nn
                    status_raw = sr
                    break

        if norm == 'detected':
            print(f"[SUCCESS] Anomaly confirmed after selecting '{t}'.")
            return t  # <-- return the winning label
        elif norm == 'none':
            print(f"[TRY {idx}] No anomaly with '{t}'. Trying next …")
            continue
        else:
            print(f"[TRY {idx}] Unknown/ambiguous status. Proceeding to next candidate …")
            continue

    print("[END] Exhausted all candidates without a confirmed anomaly.")
    return None


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
def select_by_rank(
    coords,
    heatmap_path: str,
    logger=None,
    max_distance: int = 6,
    distance_cap: int = 64,
    hold_seconds: float = 2.0,
    step_px: int = 60,      # kept for compatibility; unused in click-loop
    pause: float = 0.18,    # kept for compatibility; unused in click-loop
    read_center_status=None,
    click_fn=None,
    wait_seconds: int = 5,
    room: str | None = None,          # <— pass room in your caller
) -> str | None:                      # <— return the winning label (or None)
    """
    1) Rank labels for the given heatmap (with confidences).
    2) Try labels most likely → least likely until detection.
    3) On success: persist dhash→label in classification_map.json and copy heatmap into classifications/<room>/<label>.
    Returns the winning label, or None.
    """
    def log(msg: str):
        if callable(logger):
            logger(msg)
        else:
            print(msg)

    # Auto-wire backend.read_center_status if not provided
    if read_center_status is None:
        try:
            import backend as _be, importlib
            read_center_status = getattr(_be, "read_center_status", None)
            if read_center_status is None:
                _be = importlib.reload(_be)
                read_center_status = getattr(_be, "read_center_status", None)
        except Exception:
            read_center_status = None

    if read_center_status is None or not callable(read_center_status):
        log("read_center_status callable is required for detection loop.")
        return None

    is_known, ranked_details = _rank_labels_for_heatmap(
        heatmap_path, max_distance=max_distance, distance_cap=distance_cap
    )
    if not ranked_details:
        log("No labels found to select.")
        return None

    # Log ranking
    log("— Anomaly type ranking —")
    for item in ranked_details:
        lab, d, conf = item["label"], item["distance"], item["confidence"]
        if d is None:
            log(f"  {lab:>20}  conf={conf:.2f}  (no prior)")
        else:
            log(f"  {lab:>20}  conf={conf:.2f}  dist={d}")
    log(f"Known match within threshold: {is_known}")

    ranked_types = [(item["label"], float(item["confidence"])) for item in ranked_details]

    anchor = None

    def _move_to_label(label: str):
        nonlocal anchor
        if anchor is None:
            # First try: open menu and detect anchor, then hover label
            anchor = open_dropdown_and_anchor(coords, hold_seconds=hold_seconds, logger=logger)
            if not anchor:
                log("Failed to open dropdown / anchor; aborting selection for this label.")
                return
            ok = move_cursor_to_label_at_anchor(anchor, label, step_px=step_px, logger=logger)
            if not ok:
                log(f"Could not hover '{label}' after anchoring.")
        else:
            # Subsequent tries: re-open at coords, then hover via stored anchor
            ok = reopen_dropdown_and_hover(
                coords=coords,
                anchor=anchor,
                label=label,
                hold_seconds=hold_seconds,
                step_px=step_px,
                logger=logger
            )
            if not ok:
                log(f"Could not re-open/hover '{label}' via stored anchor.")

    # Try candidates; this should return the winning label (per earlier change)
    winning_label = select_and_click_until_detected(
        ranked_types=ranked_types,
        move_cursor_to_anomaly=_move_to_label,
        read_center_status=read_center_status,
        click_fn=click_fn,
        wait_seconds=wait_seconds,
    )

    # Persist on success
    if winning_label:
        try:
            sig = _dhash_signature(heatmap_path)
            if sig:
                save_classification_signature(sig, winning_label)
                log(f"[persist] signature saved: {sig} → {winning_label}")
            else:
                log("[persist] could not compute signature; JSON not updated.")

            if room:
                out = classify_heatmap(room, heatmap_path, winning_label)
                log(f"[persist] heatmap copied → {out}")
            else:
                log("[persist] room is None; skipped copying heatmap to classifications/")
        except Exception as e:
            log(f"[persist] error: {e}")

    return winning_label