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
# ───── CONFIG ──────────────────────────────────────────────────────
ROOMS                   = ["Yard","Entryway","Living", "Kitchen", "Bedroom", "Bathroom"  ]
BASE_DIR                = os.path.join(os.path.dirname(__file__), "LogCabin")
HEATMAP_SUBFOLDER       = "heatmaps"     # new subfolder for heatmaps
MATCH_THRESHOLD         = 0.8
BINARY_THRESH           = 20
PIXEL_COUNT_THRESHOLD = 400
CLASSIFICATION_MAP = os.path.join(BASE_DIR, "classification_map.json")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classifications")
# ────────────────────────────────────────────────────────────────────

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

def process_room():
    """
    Capture screen, detect room, and for each baseline region:
    diff that crop, then flag if pixel_count > PIXEL_COUNT_THRESHOLD.
    Heatmaps are saved under LogCabin/<Room>/heatmaps/.
    Returns (room, anomalies, None).
    """
    time.sleep(0.2)
    img = capture_screen()
    room = detect_room_name(img)
    tpl_img = template_images.get(room)
    if not room or tpl_img is None:
        return None, [], None

    heat_dir = os.path.join(BASE_DIR, room, HEATMAP_SUBFOLDER)
    os.makedirs(heat_dir, exist_ok=True)

    anomalies = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for region in baseline_regions[room]:
        cls = region["class_name"]
        x1, y1, x2, y2 = map(int, region["box"])
        live_crop = img[y1:y2, x1:x2]
        tpl_crop  = tpl_img[y1:y2, x1:x2]

        diff = cv2.absdiff(live_crop, tpl_crop)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(gray, BINARY_THRESH, 255, cv2.THRESH_BINARY)
        pix_count = int(np.count_nonzero(binm))

        if pix_count > PIXEL_COUNT_THRESHOLD:
            heat = cv2.applyColorMap(binm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(live_crop, 0.7, heat, 0.5, 0)
            # save into the heatmaps subfolder
            fname = f"{room}_{cls}_{ts}_HEAT.png"
            heat_path = os.path.join(heat_dir, fname)
            cv2.imwrite(heat_path, overlay)

            anomalies.append({
                "class_name": cls,
                "box": region["box"],
                "pixel_count": pix_count,
                "heatmap_path": heat_path
            })

    return room, anomalies, heat_path
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


# def convert_to_black_white_fullsize(heatmap_path, box=None):
#     """
#     Returns a binary image (255 = changed, 0 = unchanged).
#     If box is provided, crops to that region before processing.
#     Detects red/orange/yellow heatmap pixels and uses a 'not-blue' fallback.
#     """
#     img = cv2.imread(heatmap_path)
#     if img is None:
#         print(f"[ERROR] Unable to read heatmap: {heatmap_path}")
#         return None

#     # Crop if box is provided
#     if box:
#         x1, y1, x2, y2 = map(int, box)
#         img = img[y1:y2, x1:x2]
#         if img.size == 0:
#             print(f"[ERROR] Empty crop for {heatmap_path}")
#             return None

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # --- 1) Red ranges ---
#     red1_lo = np.array([0,   50, 40],  dtype=np.uint8)
#     red1_hi = np.array([10, 255, 255], dtype=np.uint8)
#     red2_lo = np.array([160, 50, 40],  dtype=np.uint8)
#     red2_hi = np.array([180,255, 255], dtype=np.uint8)
#     mask_red = cv2.inRange(hsv, red1_lo, red1_hi) | cv2.inRange(hsv, red2_lo, red2_hi)

#     # --- 2) Orange / Yellow ---
#     warm_lo = np.array([10,  40, 40],  dtype=np.uint8)
#     warm_hi = np.array([40, 255, 255], dtype=np.uint8)
#     mask_warm = cv2.inRange(hsv, warm_lo, warm_hi)

#     # --- 3) Not-blue fallback ---
#     sat = hsv[:, :, 1]
#     val = hsv[:, :, 2]
#     not_blue = ((hsv[:, :, 0] < 85) | (hsv[:, :, 0] > 135)) & (sat >= 40) & (val >= 40)
#     mask_not_blue = np.uint8(not_blue) * 255

#     # Combine
#     mask = cv2.bitwise_or(mask_red, mask_warm)
#     mask = cv2.bitwise_or(mask, mask_not_blue)

#     # Cleanup
#     mask = cv2.medianBlur(mask, 3)
#     kernel = np.ones((3,3), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

#     return mask
# def calculate_density_and_centroid(binary_full: np.ndarray, box=None):
#     """
#     Computes density and centroid **in full-image coordinates**.
#     If `box` is given, only pixels inside that box are considered.
#     Returns (density, (cx, cy)) where centroid is (x, y) in full image.
#     """
#     H, W = binary_full.shape[:2]

#     if box is not None:
#         x1, y1, x2, y2 = map(int, box)
#         x1 = np.clip(x1, 0, W); x2 = np.clip(x2, 0, W)
#         y1 = np.clip(y1, 0, H); y2 = np.clip(y2, 0, H)
#         roi = binary_full[y1:y2, x1:x2]
#         total_pixels = roi.size if roi.size > 0 else 1
#         ys, xs = np.where(roi > 0)
#         if len(xs) == 0:
#             return 0.0, None
#         cx = int(xs.mean()) + x1
#         cy = int(ys.mean()) + y1
#         density = len(xs) / total_pixels
#         return density, (cx, cy)

#     # whole image
#     total_pixels = binary_full.size if binary_full.size > 0 else 1
#     ys, xs = np.where(binary_full > 0)
#     if len(xs) == 0:
#         return 0.0, None
#     cx = int(xs.mean())
#     cy = int(ys.mean())
#     density = len(xs) / total_pixels
#     return density, (cx, cy)


# def get_anomaly_coordinates(anomalies, save_folder="bw_heatmaps_full"):
#     """
#     Chooses the anomaly (heatmap+box) with the highest red/orange density.
#     - Saves a FULL-SIZE black/white mask (same resolution as heatmap), with a red dot at the centroid.
#     - Returns best (cx, cy) in full-image coordinates, or None.
#     """
#     os.makedirs(save_folder, exist_ok=True)

#     best_density = -1.0
#     best_coords  = None
#     best_name    = None

#     for a in anomalies:
#         heatmap_path = a.get('heatmap_path') or a.get('heatmap')
#         box          = a.get('box')

#         if not heatmap_path or not os.path.exists(heatmap_path):
#             print(f"[ERROR] Heatmap path invalid: {heatmap_path}")
#             continue

#         binary_full = convert_to_black_white_fullsize(heatmap_path, box)
#         if binary_full is None:
#             continue

#         density, centroid = calculate_density_and_centroid(binary_full, box)
#         H, W = binary_full.shape[:2]
#         print(f"[DEBUG] {os.path.basename(heatmap_path)} | density={density:.4f} | centroid={centroid} | size=({W}x{H})")

#         # Save full-size BW with centroid
#         vis = cv2.cvtColor(binary_full, cv2.COLOR_GRAY2BGR)
#         if centroid is not None:
#             cv2.circle(vis, centroid, 6, (0, 0, 255), -1)
#         out_path = os.path.join(save_folder, os.path.basename(heatmap_path))
#         cv2.imwrite(out_path, vis)

#         if density > best_density and centroid is not None:
#             best_density = density
#             best_coords  = centroid
#             best_name    = os.path.basename(heatmap_path)

#     if best_coords is not None:
#         print(f"[FINAL] Selected {best_name} @ {best_coords} (density={best_density:.4f})")
#     else:
#         print("[FINAL] No valid coordinates found.")

#     return best_coords
def convert_to_black_white_fullsize(heatmap_path, box=None, debug_dir=None, debug_name=None):
    """
    Returns (bw_mask, used_bgr, offset_xy)
      - bw_mask: 255 = change, 0 = no change
      - used_bgr: the BGR image we actually analyzed (crop or full)
      - offset_xy: (x_off, y_off) to map local centroid to screen coords
    """
    img_full = cv2.imread(heatmap_path)
    if img_full is None:
        print(f"[ERROR] Unable to read heatmap: {heatmap_path}")
        return None, None, (0, 0)

    Hf, Wf = img_full.shape[:2]
    use_full = True
    x_off = y_off = 0

    # Try to crop if a box was provided
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = map(int, box)
        # Clamp box to current image size (heatmap might be already-cropped)
        ix1 = max(0, min(Wf, x1))
        iy1 = max(0, min(Hf, y1))
        ix2 = max(0, min(Wf, x2))
        iy2 = max(0, min(Hf, y2))

        # If the clamped box has area, use it; otherwise fall back to full heatmap
        if ix2 > ix1 and iy2 > iy1:
            cropped = img_full[iy1:iy2, ix1:ix2]
            if cropped.size > 0:
                use_full = False
                used = cropped
                x_off, y_off = x1, y1  # map local centroid back to screen coords
            else:
                print(f"[WARN] Crop empty after clamp for {os.path.basename(heatmap_path)}; using full heatmap")
                used = img_full
                x_off, y_off = x1, y1  # heatmap itself is already a crop; still offset
        else:
            print(f"[WARN] Box collapses after clamp ({box}) on {os.path.basename(heatmap_path)}; using full heatmap")
            used = img_full
            x_off, y_off = x1, y1
    else:
        used = img_full

    # Build BW mask (broad warm hue capture)
    hsv = cv2.cvtColor(used, cv2.COLOR_BGR2HSV)

    mask_red  = cv2.inRange(hsv, (0,   40,  40), (10,  255, 255)) | \
                cv2.inRange(hsv, (160, 40,  40), (180, 255, 255))
    mask_warm = cv2.inRange(hsv, (10,  30,  30), (40,  255, 255))

    b, g, r = cv2.split(used)
    mask_r = (r.astype(np.int16) - np.maximum(g, b).astype(np.int16)) > 20
    mask_y = (r > 150) & (g > 130) & (b < 170)
    mask_chan = (mask_r | mask_y).astype(np.uint8) * 255

    # Not-blue fallback (kept conservative)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    not_blue = ((h < 85) | (h > 135)) & (s >= 40) & (v >= 45)
    mask_not_blue = not_blue.astype(np.uint8) * 255

    bw = cv2.bitwise_or(mask_red | mask_warm, mask_chan)
    bw = cv2.bitwise_or(bw, mask_not_blue)

    bw = cv2.medianBlur(bw, 3)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        name = debug_name or os.path.basename(heatmap_path)
        cv2.imwrite(os.path.join(debug_dir, f"BW_{name}"), bw)

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
def get_anomaly_coordinates(anomalies, debug_dir="debug_selected"):
    """
    Pick the anomaly whose mask has the highest bright‐pixel density.
    Uses _centroid_from_mask(...) to find a robust centroid (with merge + fallback).
    Returns (screen_x, screen_y), and writes ANALYZE_* for every candidate
    and FINAL_* for the winner.
    """
    os.makedirs(debug_dir, exist_ok=True)

    best = {"density": 0.0, "coords": None, "heatmap": None, "box": None}

    for a in anomalies:
        heatmap_path = a.get("heatmap_path") or a.get("heatmap")
        box = a.get("box")

        if not heatmap_path or not os.path.exists(heatmap_path):
            print(f"[ERROR] Invalid or missing heatmap path: {heatmap_path}")
            continue

        print(f"[DEBUG] Processing: {heatmap_path}")

        # Build a full-size binary mask aligned to screen coordinates.
        bw, used_bgr, (x_off, y_off) = convert_to_black_white_fullsize(
            heatmap_path,
            box=box,
            debug_dir=debug_dir,
            debug_name=os.path.basename(heatmap_path)
        )
        if bw is None or used_bgr is None:
            continue

        # Robust centroid from mask (largest component, then merge, then global fallback).
        res = _centroid_from_mask(bw, try_merge=True)
        if not res:
            print("[DEBUG] No centroid found in this mask.")
            continue

        cx_local, cy_local, density = res
        cx_screen, cy_screen = cx_local + x_off, cy_local + y_off

        # Save candidate visualization
        comp_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        cv2.circle(comp_vis, (cx_local, cy_local), 6, (0, 0, 255), -1)
        side = cv2.hconcat([used_bgr, comp_vis])
        cv2.imwrite(os.path.join(debug_dir, f"ANALYZE_{os.path.basename(heatmap_path)}"), side)

        # Keep the best by density
        if density > best["density"]:
            best.update({
                "density": density,
                "coords": (cx_screen, cy_screen),
                "heatmap": heatmap_path,
                "box": box
            })

    if best["coords"] and best["heatmap"]:
        # Final visualization on the original heatmap image
        full = cv2.imread(best["heatmap"])
        if full is not None:
            vis = full.copy()
            if best["box"] and len(best["box"]) == 4:
                x1, y1, x2, y2 = map(int, best["box"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(vis, best["coords"], 8, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(debug_dir, f"FINAL_{os.path.basename(best['heatmap'])}"), vis)

        print(f"[RESULT] Selected {os.path.basename(best['heatmap'])}  density={best['density']:.4f}  coords={best['coords']}")
        return best["coords"]

    print("[DEBUG] No valid coordinates found.")
    return None


def move_cursor_to_anomaly(coords, hold_seconds=2, dropdown=False, logger=None):
    """
    Moves the cursor to given anomaly coordinates, performs a long click,
    and optionally tries to move to the top of the dropdown.

    Args:
        coords (tuple): (x, y) screen coordinates.
        hold_seconds (float): Time to hold mouse button down.
        dropdown (bool): If True, try to move cursor to top of dropdown after click.
        logger (callable): Optional logger function.
    """
    if not coords:
        if logger:
            logger("No coordinates provided to move_cursor_to_anomaly.")
        return False

    x, y = coords
    if logger:
        logger(f"Moving mouse to anomaly at ({x}, {y})")

    # Move to anomaly location
    pyautogui.moveTo(x, y, duration=0.2)

    # Long press
    pyautogui.mouseDown()
    time.sleep(hold_seconds)
    pyautogui.mouseUp()

    if logger:
        logger(f"Mouse held for {hold_seconds} seconds, released.")

    # Optional dropdown handling
    if dropdown:
        time.sleep(0.15)  # allow dropdown to render
        moved = move_cursor_to_dropdown_top()
        if moved and logger:
            logger("Moved to top of dropdown.")
        elif logger:
            logger("Dropdown top not found.")

    return True


def move_cursor_to_dropdown_top():
    """
    Attempts to detect the dropdown above the current cursor and move to its top.
    Returns True if successful, False otherwise.
    """
    try:
        x, y = pyautogui.position()
        left, top, right, bottom = x - 220, y - 420, x + 220, y + 60
        screen = ImageGrab.grab(bbox=(left, top, right, bottom))
        img = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        largest = max(contours, key=cv2.contourArea)
        x_c, y_c, w, h = cv2.boundingRect(largest)

        top_x_screen = left + x_c + (w // 2)
        top_y_screen = top + y_c + 5

        pyautogui.moveTo(top_x_screen, top_y_screen, duration=0.15)
        return True
    except Exception:
        return False