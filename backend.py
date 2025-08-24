import os
import time
import cv2
import mss
import numpy as np
import easyocr
from datetime import datetime
import json
import shutil
import pyautogui
from PIL import ImageGrab
import cv2
import tempfile
# ───── CONFIG ──────────────────────────────────────────────────────
ROOMS                   = ["Yard","Entryway","Living", "Kitchen", "Bedroom", "Bathroom"  ]
BASE_DIR                = os.path.join(os.path.dirname(__file__), "LogCabin")
HEATMAP_SUBFOLDER       = "heatmaps"     # new subfolder for heatmaps
MATCH_THRESHOLD         = 0.8
BINARY_THRESH           = 20
PIXEL_COUNT_THRESHOLD = 400
CLASSIFICATION_MAP = os.path.join(BASE_DIR, "classification_map.json")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classifications")
DEBUG_PERSIST = True  

pyautogui.FAILSAFE = False

# ────────────────────────────────────────────────────────────────────
ANOMALY_POSITIONS = {
    "Dead body": 1,
    "Door anomaly": 2,
    "Extra object": 3,
    "Image anomaly": 4,
    "Intruder": 5,
    "Missing Object": 6,
    "Object Manipulation": 7,
    "Object Movement": 8,
    "Object Replacement": 9,
    "Other": 10,
}



def set_debug_persistence(enabled: bool):
    """Turn keeping debug artifacts (heatmaps & BW) on/off globally."""
    global DEBUG_PERSIST
    DEBUG_PERSIST = bool(enabled)

def is_debug_persistence_enabled() -> bool:
    return DEBUG_PERSIST

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load room templates
template_images = {
    room: cv2.imread(os.path.join(BASE_DIR, room, "template.png"))
    for room in ROOMS
}
if os.path.exists(CLASSIFICATION_MAP):
    with open(CLASSIFICATION_MAP, "r") as f:
        classification_map = json.load(f)
else:
    classification_map = {}
# Pre-load cropped regions per room
group_templates = {}
for room in ROOMS:
    tpl_dir = os.path.join(BASE_DIR, room, "group_templates")
    crops = {}
    if os.path.isdir(tpl_dir):
        for fn in os.listdir(tpl_dir):
            name, ext = os.path.splitext(fn)
            if ext.lower() == ".png":
                path = os.path.join(tpl_dir, fn)
                crops[name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    group_templates[room] = crops



def capture_screen():
    with mss.mss() as sct:
        frame = np.array(sct.grab(sct.monitors[1]))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def detect_room_name(img):
    h, w = img.shape[:2]
    roi = img[int(h*0.80):h, 0:int(w*0.30)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    for _, text, _ in reader.readtext(gray):
        for room in ROOMS:
            if room.lower() in text.lower():
                return room
    return None

def detect_regions_in_template(room):
    tpl_img = template_images.get(room)
    if tpl_img is None:
        return []
    gray_tpl = cv2.cvtColor(tpl_img, cv2.COLOR_BGR2GRAY)
    regs = []
    for name, tpl in group_templates[room].items():
        if tpl is None:
            continue
        res = cv2.matchTemplate(gray_tpl, tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= MATCH_THRESHOLD:
            x, y = max_loc
            h, w = tpl.shape
            regs.append({
                "class_name": name,
                "box": [float(x), float(y), float(x+w), float(y+h)]
            })
    return regs

# Precompute baseline regions
baseline_regions = {
    room: detect_regions_in_template(room)
    for room in ROOMS
}

def process_room(persist: bool | None = None):
    """
    Capture screen, detect room, diff the known regions against the room template,
    and save a WHITE-overlay heatmap plus its exact binary mask.

    Returns:
        (room_name: str | None, anomalies: list[dict], last_heatmap_path: str | None)

    Each anomaly dict contains:
        - class_name: str
        - box: [x1, y1, x2, y2]
        - pixel_count: int            # count of white pixels in mask
        - heatmap_path: str           # path to saved overlay image (live + white)
        - mask_path: str              # path to saved binary mask (0/255)
        - bw_mask: np.ndarray         # in-memory binary mask (H×W, uint8)
        - baseline: np.ndarray        # template crop (BGR)
        - current:  np.ndarray        # live crop (BGR)
        - transient: bool             # true if written to a temp dir
        - transient_root: str | None  # the temp dir root to allow cleanup
    """
    import tempfile
    from datetime import datetime

    if persist is None:
        persist = DEBUG_PERSIST

    # 1) Capture + room detection
    time.sleep(0.2)
    img = capture_screen()
    room = detect_room_name(img)
    tpl_img = template_images.get(room)
    if not room or tpl_img is None:
        return None, [], None

    # 2) Choose output directory
    if persist:
        heat_dir = os.path.join(BASE_DIR, room, HEATMAP_SUBFOLDER)
        os.makedirs(heat_dir, exist_ok=True)
        transient_root = None
    else:
        # ephemeral artifacts; caller can clean via _cleanup_transient_artifacts
        heat_dir = tempfile.mkdtemp(prefix=f"{room}_heatmaps_tmp_", dir=os.path.join(BASE_DIR, room))
        transient_root = heat_dir
        os.makedirs(heat_dir, exist_ok=True)

    anomalies = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_heat_path = None

    # 3) Iterate known regions and compute abs-diff mask
    for region in baseline_regions.get(room, []):
        cls = region["class_name"]
        x1, y1, x2, y2 = map(int, region["box"])

        live_crop = img[y1:y2, x1:x2]
        tpl_crop  = tpl_img[y1:y2, x1:x2]
        if live_crop.size == 0 or tpl_crop.size == 0:
            continue

        # abs-diff → grayscale → binary mask
        diff = cv2.absdiff(live_crop, tpl_crop)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # (optional) light blur helps stabilize noise; keep small to preserve edges
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binm = cv2.threshold(gray, BINARY_THRESH, 255, cv2.THRESH_BINARY)

        pix_count = int(np.count_nonzero(binm))
        if pix_count <= PIXEL_COUNT_THRESHOLD:
            continue

        # 4) Build WHITE overlay (live + white mask)
        white_bgr = cv2.cvtColor(binm, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(live_crop, 0.7, white_bgr, 0.5, 0)

        # 5) Persist overlay and the raw mask
        base = f"{room}_{cls}_{ts}"
        heat_path = os.path.join(heat_dir, f"{base}_HEAT.png")
        mask_path = os.path.join(heat_dir, f"{base}_MASK.png")

        cv2.imwrite(heat_path, overlay)
        cv2.imwrite(mask_path, binm)

        last_heat_path = heat_path

        # 6) Collect anomaly record (keep mask + crops in memory too)
        anomalies.append({
            "class_name": cls,
            "box": region["box"],
            "pixel_count": pix_count,
            "heatmap_path": heat_path,
            "mask_path": mask_path,
            "bw_mask": binm.copy(),     # for direct centroiding
            "baseline": tpl_crop,       # template crop (BGR)
            "current":  live_crop,      # live crop (BGR)
            "transient": not persist,
            "transient_root": transient_root
        })

    return room, anomalies, last_heat_path


def save_classification_signature(signature: str, anomaly_type: str):
    classification_map[signature] = anomaly_type
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(CLASSIFICATION_MAP, "w") as f:
        json.dump(classification_map, f, indent=2)

def classify_heatmap(room: str, heatmap_path: str, anomaly_type: str):
    if not anomaly_type or not os.path.exists(heatmap_path):
        raise ValueError("Invalid anomaly type or heatmap path")

    dest_dir = os.path.join(CLASSIFICATION_DIR, room, anomaly_type)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(heatmap_path, dest_dir)
    return os.path.join(dest_dir, os.path.basename(heatmap_path))
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
# UNDER WORK IN PROGRESS
def _cleanup_transient_artifacts(anomalies):
    """
    Remove temporary heatmap folders/files produced by process_room(persist=False).
    Safely ignores missing paths.
    """
    # Prefer removing the temp root folder when provided
    transient_roots = set()
    to_remove_files = []

    for a in anomalies:
        if a.get("transient_root"):
            transient_roots.add(a["transient_root"])
        elif a.get("transient") and a.get("heatmap_path"):
            to_remove_files.append(a["heatmap_path"])

    # Remove temp directories
    for root in transient_roots:
        try:
            if os.path.isdir(root):
                shutil.rmtree(root, ignore_errors=True)
        except Exception as e:
            print(f"[WARN] Failed to remove transient root '{root}': {e}")

    # Remove stray files if any
    for f in to_remove_files:
        try:
            if os.path.isfile(f):
                os.remove(f)
        except Exception as e:
            print(f"[WARN] Failed to remove transient file '{f}': {e}")


def convert_to_black_white_fullsize(
    heatmap_path: str,
    box=None,
    debug_dir: str | None = None,
    debug_name: str | None = None,
    *,
    precomputed_mask: np.ndarray | None = None,
    baseline_bgr: np.ndarray | None = None,
    current_bgr:  np.ndarray | None = None,
    diff_blur_ksize: int = 3,
    diff_thresh: int | str = "auto",
    morph_open: int = 1,
    morph_close: int = 2,
    min_white_frac: float = 2e-4,
    max_white_frac: float = 0.25,
):
    def _dump(tag: str, mat: np.ndarray):
        if not (DEBUG_PERSIST and debug_dir):
            return
        os.makedirs(debug_dir, exist_ok=True)
        nm = debug_name or os.path.basename(heatmap_path)
        cv2.imwrite(os.path.join(debug_dir, f"{tag}_{nm}"), mat)

    # --- compute region offset ONCE ---
    x_off = y_off = 0
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        x_off, y_off = x1, y1

    # --- 0) precomputed mask path ---
    if precomputed_mask is not None:
        bw = precomputed_mask
        _dump("MASK_FROM_CALLER", bw)

        # Provide a visual counterpart for debug panes
        if current_bgr is not None:
            used_bgr = current_bgr
        else:
            used_full = cv2.imread(heatmap_path)
            if used_full is not None and box is not None and len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                Hf, Wf = used_full.shape[:2]
                ix1 = max(0, min(Wf, x1)); iy1 = max(0, min(Hf, y1))
                ix2 = max(0, min(Wf, x2)); iy2 = max(0, min(Hf, y2))
                used_bgr = used_full[iy1:iy2, ix1:ix2] if (ix2 > ix1 and iy2 > iy1) else used_full
            else:
                used_bgr = used_full  # may be None; that's okay

        # IMPORTANT: return the REAL offset, not (0,0)
        return bw, used_bgr, (x_off, y_off)

    # --- determine region offset if a box was provided ---
    x_off = y_off = 0
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        x_off, y_off = x1, y1

    # --- 1) Preferred: abs-diff if baseline/current are available ---
    if baseline_bgr is not None and current_bgr is not None:
        try:
            base = baseline_bgr
            curr = current_bgr

            # size match to each other (they should already be same in your pipeline)
            if base.shape[:2] != curr.shape[:2]:
                target = (curr.shape[1], curr.shape[0])
                base = cv2.resize(base, target, interpolation=cv2.INTER_AREA)

            base_g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            # light blur to suppress sensor/game noise
            k = max(1, int(diff_blur_ksize))
            if k % 2 == 0:
                k += 1
            base_g = cv2.GaussianBlur(base_g, (k, k), 0)
            curr_g = cv2.GaussianBlur(curr_g, (k, k), 0)

            diff = cv2.absdiff(base_g, curr_g)
            _dump("D0_DIFF", diff)

            if isinstance(diff_thresh, str) and diff_thresh.lower() == "auto":
                # Otsu first
                tval, bw = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # fall back to a percentile if Otsu is too low (very flat images)
                if tval < 12:
                    p = np.percentile(diff, 85)  # 80–90 works well; adjust if needed
                    t = max(8, min(64, int(p)))
                    _, bw = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
            else:
                t = int(diff_thresh)
                _, bw = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

            _dump("D1_THRESH", bw)

            # morphology
            k3 = np.ones((3, 3), np.uint8)
            if morph_open > 0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=morph_open)
            if morph_close > 0:
                bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=morph_close)

            # guardrails
            frac = float(cv2.countNonZero(bw)) / float(bw.size)
            if frac < min_white_frac:
                bw = cv2.dilate(bw, k3, iterations=1)
                _dump("D2_DILATE_SPARSE", bw)
            elif frac > max_white_frac:
                bw = cv2.erode(bw, k3, iterations=1)
                _dump("D2_ERODE_DENSE", bw)

            _dump("D3_FINAL", bw)
            return bw, curr, (x_off, y_off)

        except Exception as e:
            print(f"[WARN] absdiff path failed, will try fallback: {e}")

    # --- 2) Fallback: try to recover white from the saved heatmap image ---
    # Your new heatmaps are live_crop with white painted where diff==1,
    # so a bright-threshold usually isolates the change.
    img_full = cv2.imread(heatmap_path)
    if img_full is None:
        print(f"[ERROR] Unable to read heatmap: {heatmap_path}")
        return None, None, (0, 0)

    used = img_full
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        Hf, Wf = img_full.shape[:2]
        ix1 = max(0, min(Wf, x1)); iy1 = max(0, min(Hf, y1))
        ix2 = max(0, min(Wf, x2)); iy2 = max(0, min(Hf, y2))
        if ix2 > ix1 and iy2 > iy1:
            used = img_full[iy1:iy2, ix1:ix2]

    gray = cv2.cvtColor(used, cv2.COLOR_BGR2GRAY)
    # bright-threshold; use Otsu then clamp to high side
    t_otsu, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    t = max(t_otsu, 200)  # ensure we really pick only bright whites
    _, bw = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    # small tidy
    k3 = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=1)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k3, iterations=1)

    _dump("FALLBACK_FINAL", bw)
    return bw, used, (x_off, y_off)


def _centroid_from_mask(bw: np.ndarray, *, try_merge=True):
    """
    Returns (cx, cy, density) in local (mask) coordinates.
    - try_merge: if True, dilate & retry once when no valid component exists.
    density = largest_component_area / mask_area (or fraction of all white pixels in fallback)
    """
    H, W = bw.shape[:2]
    area_img = H * W

    def largest_component_centroid(mask):
        num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None
        # min-area threshold: small but non-zero
        min_area = max(20, int(0.00015 * area_img))
        best_lbl, best_area = None, -1
        for lbl in range(1, num):
            a = stats[lbl, cv2.CC_STAT_AREA]
            if a >= min_area and a > best_area:
                best_lbl, best_area = lbl, a
        if best_lbl is None:
            return None
        cx, cy = map(int, np.round(cents[best_lbl]))
        density = best_area / float(area_img)
        return cx, cy, density

    # 1) try straight components
    out = largest_component_centroid(bw)
    if out:
        return out

    # 2) optional merge pass (to connect speckles)
    if try_merge:
        merged = cv2.dilate(bw, np.ones((5, 5), np.uint8), iterations=2)
        out = largest_component_centroid(merged)
        if out:
            return out

    # 3) ultimate fallback: centroid of all white pixels
    ys, xs = np.where(bw > 0)
    if len(xs):
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))
        density = len(xs) / float(area_img)  # fraction white
        return cx, cy, density

    return None
def get_anomaly_coordinates(anomalies, debug_dir="debug_selected", persist: bool | None = None):
    """
    Pick the anomaly whose mask has the highest bright‐pixel density.
    Uses _centroid_from_mask(...) to find a robust centroid (merge + fallback).
    Returns (screen_x, screen_y).

    Debug images (ANALYZE_*/FINAL_*) are only written when persistence is enabled.
    If persistence is disabled, transient heatmaps created by process_room() are deleted.
    """
    if persist is None:
        persist = DEBUG_PERSIST

    # Only allocate a real debug dir when persisting
    local_debug_dir = debug_dir if persist else None
    if persist:
        os.makedirs(local_debug_dir, exist_ok=True)

    best = {"density": 0.0, "coords": None, "heatmap": None, "box": None}

    for a in anomalies:
        heatmap_path = a.get("heatmap_path") or a.get("heatmap")
        box = a.get("box")

        if not heatmap_path or not os.path.exists(heatmap_path):
            print(f"[ERROR] Invalid or missing heatmap path: {heatmap_path}")
            continue

        print(f"[DEBUG] Processing: {heatmap_path}")

        bw, used_bgr, (x_off, y_off) = convert_to_black_white_fullsize(
    heatmap_path=heatmap_path,          # <-- use the resolved variable, not a["heatmap_path"]
    box=box,
    precomputed_mask=a.get("bw_mask"),  # fastest path
    baseline_bgr=a.get("baseline"),
    current_bgr=a.get("current"),
    debug_dir=local_debug_dir,
    debug_name=os.path.basename(heatmap_path),
)
        if bw is None:
            continue

        res = _centroid_from_mask(bw, try_merge=True)
        if not res:
            print("[DEBUG] No centroid found in this mask.")
            continue

        cx_local, cy_local, density = res
        cx_screen, cy_screen = cx_local + x_off, cy_local + y_off

        # Candidate visualization only if persisting
        if persist and local_debug_dir:
            comp_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            cv2.circle(comp_vis, (cx_local, cy_local), 6, (0, 0, 255), -1)
            side = cv2.hconcat([used_bgr, comp_vis])
            cv2.imwrite(os.path.join(local_debug_dir, f"ANALYZE_{os.path.basename(heatmap_path)}"), side)

        if density > best["density"]:
            best.update({
                "density": density,
                "coords": (cx_screen, cy_screen),
                "heatmap": heatmap_path,
                "box": box
            })

    # Final visualize + optional cleanup
    if best["coords"] and best["heatmap"]:
        if persist and local_debug_dir:
            full = cv2.imread(best["heatmap"])
            if full is not None:
                vis = full.copy()
                if best["box"] and len(best["box"]) == 4:
                    x1, y1, x2, y2 = map(int, best["box"])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(vis, best["coords"], 8, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(local_debug_dir, f"FINAL_{os.path.basename(best['heatmap'])}"), vis)

        print(f"[RESULT] Selected {os.path.basename(best['heatmap'])}  density={best['density']:.4f}  coords={best['coords']}")

        # If persistence is OFF, clean transient artifacts (temp dirs from process_room)
        if not persist:
            _cleanup_transient_artifacts(anomalies)

        return best["coords"]

    print("[DEBUG] No valid coordinates found.")
    if not persist:
        _cleanup_transient_artifacts(anomalies)
    return None
# UNDER WORK
# UNDER WORK
# UNDER WORK
# UNDER WORK
# UNDER WORK
# UNDER WORK
# --- backend.py additions/updates ---

def move_cursor_to_dropdown_top_any_side(
    logger=None,
    delta_to_top: int = 290,         # pixels from "Intruder" row to top row
    nudge_px: int = 12,              # small push inside the top row
    search_pad_xy: tuple[int, int] = (380, 260),  # ROI half-width/height around cursor
    upscale: float = 1.45,           # modest upscale for OCR speed/accuracy
    delay_seconds: float = 0.0       # keep 0.0 in production; set 2.0 if you want the test behavior
) -> tuple[int, int] | None:
    """
    Anchor to the top of the anomaly dropdown by locating the word 'Intruder'
    near the cursor (EasyOCR), then jumping ~delta_to_top pixels above it.

    Returns (ax, ay) absolute screen coords on success, or None on failure.
    """
    try:
        if delay_seconds > 0:
            time.sleep(delay_seconds)  # optional: mimic test script timing

        # --- ROI around cursor ---
        sw, sh = pyautogui.size()
        cx, cy = pyautogui.position()
        pad_x, pad_y = search_pad_xy
        left   = max(0, cx - pad_x)
        top    = max(0, cy - pad_y)
        right  = min(sw, cx + pad_x)
        bottom = min(sh, cy + pad_y)
        if right - left < 40 or bottom - top < 40:
            if logger: logger("Anchor ROI too small.")
            return None

        # Capture ROI
        roi_rgb = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
        roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

        # --- OCR helpers (reuse backend.reader) ---
        def _prep_fast(gray: np.ndarray):
            norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            blur = cv2.GaussianBlur(norm, (3, 3), 0)
            _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return norm, bw

        def _is_intruder(txt: str) -> bool:
            if not txt: return False
            t = "".join(ch.lower() for ch in txt if ch.isalnum())
            return "intrud" in t  # fuzzy substring

        # Upscale once to help EasyOCR on UI fonts
        if upscale and upscale > 1.0:
            roi_gray_up = cv2.resize(roi_gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        else:
            roi_gray_up = roi_gray

        norm, bw = _prep_fast(roi_gray_up)
        inv = 255 - bw

        def _ocr_once(arr):
            # Use your initialized reader for speed
            return reader.readtext(
                arr,
                detail=1,
                paragraph=False,
                decoder="greedy",     # faster
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.5,
                low_text=0.2,
            )

        # Try normalized first, then quick inverted fallback
        best = None  # (conf, cx_local, cy_local, bbox_points)
        for variant in (norm, inv):
            try:
                results = _ocr_once(variant)
            except Exception:
                results = []
            for bbox, text, conf in results:
                if not isinstance(text, str) or not text.strip():
                    continue
                if _is_intruder(text):
                    xs = [int(p[0]) for p in bbox]
                    ys = [int(p[1]) for p in bbox]
                    cx_local = (min(xs) + max(xs)) // 2
                    cy_local = (min(ys) + max(ys)) // 2
                    if best is None or conf > best[0]:
                        best = (conf, cx_local, cy_local, bbox)
            if best is not None:
                break  # found it; no need for further variants

        if best is None:
            if logger: logger("EasyOCR couldn't find 'Intruder' near cursor.")
            return None

        conf, cx_local_up, cy_local_up, bbox_up = best

        # Convert back if we upscaled
        inv_scale = 1.0 / (upscale if upscale else 1.0)
        cx_local = int(cx_local_up * inv_scale)
        cy_local = int(cy_local_up * inv_scale)

        # Absolute screen coords of detected word center
        word_cx_screen = left + cx_local
        word_cy_screen = top  + cy_local

        # Compute top row anchor
        anchor_x = int(np.clip(word_cx_screen, 0, sw - 1))
        anchor_y = int(np.clip(word_cy_screen - delta_to_top + nudge_px, 0, sh - 1))

        pyautogui.moveTo(anchor_x, anchor_y, duration=0.08)

        if logger:
            logger(f"Anchor via 'Intruder' at conf={conf:.2f}; top≈({anchor_x},{anchor_y}) "
                   f"using Δ={delta_to_top}, nudge={nudge_px}")

        # --- Debug overlay (saved only if DEBUG_PERSIST) ---
        if DEBUG_PERSIST:
            try:
                dbg = roi_rgb.copy()
                # scale bbox back down if needed
                pts = np.array([ (int(p[0]*inv_scale), int(p[1]*inv_scale)) for p in bbox_up ], dtype=np.int32)
                # Draw word bbox (red) and center (green)
                cv2.polylines(dbg, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
                cv2.circle(dbg, (cx_local, cy_local), 6, (0, 255, 0), -1)
                # Draw computed top-row point (blue)
                top_local_x = anchor_x - left
                top_local_y = anchor_y - top
                cv2.circle(dbg, (top_local_x, top_local_y), 6, (255, 0, 0), -1)

                os.makedirs(os.path.join(BASE_DIR, "debug_dropdown"), exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # out_path = os.path.join(BASE_DIR, "debug_dropdown", f"intruder_anchor_{ts}.png")
                # cv2.imwrite(out_path, cv2.cvtColor(dbg, cv2.COLOR_RGB2BGR))
                # if logger: logger(f"[debug] saved overlay → {out_path}")
            except Exception as e:
                if logger: logger(f"[debug save] {e}")

        return (anchor_x, anchor_y)

    except Exception as e:
        if logger:
            logger(f"[dropdown anchor via 'Intruder'] {e}")
        return None



def move_cursor_to_anomaly(coords, hold_seconds=2, dropdown=False, anomaly_type="Extra Object", logger=None):
    if not coords:
        if logger: logger("No coordinates provided to move_cursor_to_anomaly.")
        return False

    x, y = coords
    if logger: logger(f"[cursor] long-press at anomaly coords ({x},{y}) for {hold_seconds:.2f}s")
    pyautogui.moveTo(x, y, duration=0.2)
    pyautogui.mouseDown(); time.sleep(hold_seconds); pyautogui.mouseUp()

    if dropdown:
        time.sleep(0.15)
        anchor = move_cursor_to_dropdown_top_any_side(logger=logger)
        if anchor:
            if anomaly_type:
                idx = ANOMALY_POSITIONS.get(anomaly_type)
                if idx is not None:
                    ax, ay = anchor
                    sw, sh = pyautogui.size()
                    dest_x, dest_y = ax, ay + idx * 60
                    dest_x = max(0, min(sw - 1, dest_x))
                    dest_y = max(0, min(sh - 1, dest_y))
                    if logger:
                        logger(f"[cursor] hover '{anomaly_type}' idx={idx} → ({dest_x},{dest_y}) "
                               f"[anchor=({ax},{ay}) screen={sw}x{sh}]")
                    pyautogui.moveTo(dest_x, dest_y, duration=0.12)
                else:
                    if logger: logger(f"Unknown anomaly type: {anomaly_type}")
            else:
                if logger: logger("No anomaly type provided; staying at top.")
        else:
            if logger: logger("Failed to anchor dropdown; leaving cursor in place.")
    return True

# --- NEW HELPERS ---
from typing import Optional, Tuple


def open_dropdown_and_anchor(coords, hold_seconds: float = 2.0, logger=None):
    if not coords:
        if logger: logger("open_dropdown_and_anchor: coords missing.")
        return None
    x, y = coords
    try:
        if logger: logger(f"[cursor] long-press at anomaly coords ({x},{y}) for {hold_seconds:.2f}s")
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.mouseDown(); time.sleep(hold_seconds); pyautogui.mouseUp()
        time.sleep(0.15)
        anchor = move_cursor_to_dropdown_top_any_side(logger=logger)
        if not anchor:
            if logger: logger("[cursor] open_dropdown_and_anchor: failed to find dropdown anchor")
            return None
        if logger: logger(f"[anchor] dropdown top-row anchor set to {anchor}")
        return anchor
    except Exception as e:
        if logger: logger(f"open_dropdown_and_anchor error: {e}")
        return None

def move_cursor_to_label_at_anchor(anchor, label: str, step_px: int = 60, logger=None) -> bool:
    if not anchor:
        if logger: logger("move_cursor_to_label_at_anchor: anchor missing.")
        return False
    idx = ANOMALY_POSITIONS.get(label)
    if idx is None:
        if logger: logger(f"move_cursor_to_label_at_anchor: unknown label '{label}'")
        return False
    try:
        ax, ay = anchor
        sw, sh = pyautogui.size()
        dest_x, dest_y = ax, ay + idx * step_px
        dest_x = max(0, min(sw - 1, dest_x))
        dest_y = max(0, min(sh - 1, dest_y))
        if logger:
            logger(f"[cursor] hover '{label}' idx={idx} step={step_px} → ({dest_x},{dest_y}) "
                   f"[anchor=({ax},{ay}) screen={sw}x{sh}]")
        pyautogui.moveTo(dest_x, dest_y, duration=0.12)
        return True
    except Exception as e:
        if logger: logger(f"move_cursor_to_label_at_anchor error: {e}")
        return False

def reopen_dropdown_and_hover(coords, anchor, label, hold_seconds: float = 2.0, step_px: int = 60, logger=None) -> bool:
    if not coords or not anchor:
        if logger: logger("reopen_dropdown_and_hover: coords/anchor missing.")
        return False
    try:
        x, y = coords
        if logger: logger(f"[cursor] re-open dropdown: long-press at ({x},{y}) for {hold_seconds:.2f}s")
        pyautogui.moveTo(x, y, duration=0.2)
        pyautogui.mouseDown(); time.sleep(hold_seconds); pyautogui.mouseUp()
        time.sleep(0.12)
        return move_cursor_to_label_at_anchor(anchor, label, step_px=step_px, logger=logger)
    except Exception as e:
        if logger: logger(f"reopen_dropdown_and_hover error: {e}")
        return False

# --- Center-status OCR with debug overlay ------------------------------------
def read_center_status(
    logger=None,
    *,
    roi_rel=(0.32, 0.40, 0.68, 0.60),  # (x1_rel, y1_rel, x2_rel, y2_rel) center box
    upscale=1.6,
    save_debug=True,
    debug_dir_name="debug_center_status",
):
    """
    OCR the center of the screen for status text and (optionally) save a debug image.

    Returns the raw OCR text (joined lines). Selector will interpret it.
    """
    try:
        # 1) Grab full screen, crop center ROI (relative box)
        frame = capture_screen()                        # BGR
        H, W = frame.shape[:2]
        x1 = max(0, int(W * roi_rel[0])); y1 = max(0, int(H * roi_rel[1]))
        x2 = min(W, int(W * roi_rel[2])); y2 = min(H, int(H * roi_rel[3]))
        roi_bgr = frame[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            if logger: logger("[center OCR] ROI empty")
            return ""

        # 2) Prep for OCR
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        if upscale and upscale > 1.0:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        # Light normalization + binarized variant
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(norm, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3) Run EasyOCR (norm first, then inverted BW as fallback)
        def ocr(arr):
            try:
                return reader.readtext(
                    arr,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                )
            except Exception as e:
                if logger: logger(f"[center OCR] {e}")
                return []

        results = ocr(norm)
        if not results:
            results = ocr(255 - bw)

        # 4) Collect raw text + prepare debug overlay
        raw_lines = []
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # work in upscaled ROI space
        best_hits = []  # boxes that match either phrase

        def _norm(t: str) -> str:
            return "".join(ch.lower() for ch in (t or "")).strip()

        for bbox, text, conf in results:
            if not text:
                continue
            raw_lines.append(text)
            pts = np.array(bbox, dtype=np.int32)
            # Color code hits vs. others
            t_norm = _norm(text)
            hit = (("anomaly" in t_norm and "detected" in t_norm) or
                   ("no" in t_norm and "anomal" in t_norm))
            color = (0, 0, 255) if hit else (60, 160, 255)  # red for hits
            cv2.polylines(overlay, [pts], True, color, 2)
            (tx, ty) = (int(pts[0][0]), int(pts[0][1]) - 4)
            cv2.putText(overlay, f"{text} ({conf:.2f})", (tx, max(12, ty)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            if hit:
                best_hits.append((conf, pts, text))

        # 5) Save debug (ROI + overlay placed back on full-screen box)
        if save_debug and DEBUG_PERSIST:
            try:
                dbg_dir = os.path.join(BASE_DIR, debug_dir_name)
                os.makedirs(dbg_dir, exist_ok=True)

                # Downscale overlay back to ROI size if we upscaled
                if upscale and upscale > 1.0:
                    overlay_small = cv2.resize(
                        overlay, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA
                    )
                else:
                    overlay_small = overlay

                # Compose full-frame preview with ROI rectangle and overlay pasted
                preview = frame.copy()
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
                preview[y1:y2, x1:x2] = overlay_small

                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                # Label the file with a quick guess of interpretation
                joined_lower = " ".join(raw_lines).lower()
                if "anomaly" in joined_lower and "detected" in joined_lower:
                    tag = "detected"
                elif "no" in joined_lower and "anomal" in joined_lower:
                    tag = "none"
                else:
                    tag = "unknown"

                out_path = os.path.join(dbg_dir, f"center_ocr_{tag}_{ts}.png")
                cv2.imwrite(out_path, preview)
                if logger: logger(f"[center OCR debug] saved → {out_path}")
            except Exception as e:
                if logger: logger(f"[center OCR debug save] {e}")

        # 6) Return what selector expects (raw text)
        return " ".join(raw_lines)

    except Exception as e:
        if logger: logger(f"[center OCR error] {e}")
        return ""
