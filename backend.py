import os
import time
import cv2
import mss
import numpy as np
import easyocr
from datetime import datetime
import json
import shutil

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

# def mask_dynamic(img):
#     h, w = img.shape[:2]
#     m = img.copy()
#     cv2.rectangle(m, (0, int(h*0.80)), (int(w*0.30), h), (0,0,0), -1)
#     return m

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

    # mask dynamic UI
    # img_m = mask_dynamic(img)
    # tpl_m = mask_dynamic(tpl_img)

    # ensure the heatmap directory exists
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


# def get_anomaly_coordinates(anomalies):
#     best_coords = None
#     highest_density = 0
#     best_image_name = None

#     bw_folder = os.path.join(os.getcwd(), "bw_heatmaps")
#     os.makedirs(bw_folder, exist_ok=True)

#     for anomaly in anomalies:
#         heatmap_path = anomaly.get('heatmap_path')
#         box = anomaly.get('box')

#         if not heatmap_path or not os.path.exists(heatmap_path):
#             print(f"[ERROR] Heatmap path invalid: {heatmap_path}")
#             continue

#         x1, y1, x2, y2 = map(int, box)

#         heatmap = cv2.imread(heatmap_path)
#         if heatmap is None:
#             print(f"[ERROR] Failed to read heatmap: {heatmap_path}")
#             continue

#         cropped = heatmap[y1:y2, x1:x2]
#         if cropped.size == 0:
#             print(f"[ERROR] Empty crop from: {heatmap_path}")
#             continue

#         gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#         blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Otsu thresholding for better binary segmentation
#         _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         bright_pixels = cv2.countNonZero(binary)
#         total_pixels = binary.size
#         density = bright_pixels / total_pixels

#         print(f"[DEBUG] Processing anomaly with heatmap: {heatmap_path}")
#         print(f"[DEBUG] Density: {density:.4f}, Bright pixels: {bright_pixels}, Total: {total_pixels}")

#         # Find contours to calculate centroid
#         contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             continue

#         # Merge all contours to find total centroid
#         all_points = np.vstack(contours)
#         M = cv2.moments(all_points)
#         if M["m00"] != 0:
#             cx = int(M["m10"] / M["m00"]) + x1
#             cy = int(M["m01"] / M["m00"]) + y1
#         else:
#             cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2  # Fallback to center

#         # Update best
#         if density > highest_density:
#             highest_density = density
#             best_coords = (cx, cy)
#             best_image_name = os.path.basename(heatmap_path)

#         # Draw centroid on black-and-white image and save
#         color_output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
#         cv2.circle(color_output, (cx - x1, cy - y1), 5, (0, 0, 255), -1)
#         save_path = os.path.join(bw_folder, os.path.basename(heatmap_path))
#         cv2.imwrite(save_path, color_output)

#     if best_coords:
#         print(f"[FINAL] Selected coordinates: {best_coords} from heatmap: {best_image_name}")
#     else:
#         print("[DEBUG] No valid coordinates found.")

#     return best_coords


def convert_to_black_white(heatmap_path, box, save_dir="bw_heatmaps"):
    import cv2, numpy as np, os

    os.makedirs(save_dir, exist_ok=True)
    x1, y1, x2, y2 = map(int, box)

    heatmap = cv2.imread(heatmap_path)
    if heatmap is None:
        print(f"[ERROR] Failed to read heatmap: {heatmap_path}")
        return None, 0, None

    cropped = heatmap[y1:y2, x1:x2]
    if cropped.size == 0:
        print(f"[ERROR] Empty crop from: {heatmap_path}")
        return None, 0, None

    # Convert to HSV and mask only red/orange pixels
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    # Orange to Red - expanded hue and lower saturation/value thresholds
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([15, 255, 255])

    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    binary = red_mask  # Already a binary mask
    bright_pixels = cv2.countNonZero(binary)
    total_pixels = binary.size
    density = bright_pixels / total_pixels

    # Get contours and centroid
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, density, None

    all_points = np.vstack(contours)
    M = cv2.moments(all_points)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
    else:
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

    # Save output with red dot on centroid
    output_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.circle(output_img, (cx - x1, cy - y1), 5, (0, 0, 255), -1)
    save_path = os.path.join(save_dir, os.path.basename(heatmap_path))
    cv2.imwrite(save_path, output_img)

    return (cx, cy), density, save_path


def get_anomaly_coordinates(anomalies):
    best_coords = None
    highest_density = 0
    best_image_name = None

    for anomaly in anomalies:
        heatmap_path = anomaly.get('heatmap_path')
        box = anomaly.get('box')

        if not heatmap_path or not os.path.exists(heatmap_path):
            print(f"[ERROR] Heatmap path invalid: {heatmap_path}")
            continue

        coords, density, saved_path = convert_to_black_white(heatmap_path, box)

        print(f"[DEBUG] Processing: {os.path.basename(heatmap_path)}")
        print(f"[DEBUG] Density: {density:.4f}, Centroid: {coords}, Saved: {saved_path}")

        if coords and density > highest_density:
            highest_density = density
            best_coords = coords
            best_image_name = os.path.basename(heatmap_path)

    if best_coords:
        print(f"[FINAL] Selected coordinates: {best_coords} from heatmap: {best_image_name}")
    else:
        print("[DEBUG] No valid coordinates found.")

    return best_coords