import os
import time
import cv2
import mss
import numpy as np
import easyocr
from datetime import datetime

# ---------- CONFIG ----------
ROOMS                   = ["Living", "Kitchen", "Bedroom", "Bathroom", "Entryway", "Yard"]
BASE_DIR                = os.path.join(os.path.dirname(__file__), "LogCabin")
MATCH_THRESHOLD         = 0.8   # template-match confidence
BINARY_THRESH           = 30    # increased to ignore more subtle shifts
PIXEL_COUNT_THRESHOLD   = 1000  # increased to ignore small changes
# -----------------------------

# Initialize OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load room templates
template_images = {
    room: cv2.imread(os.path.join(BASE_DIR, room, "template.png"))
    for room in ROOMS
}

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

def mask_dynamic(img):
    h, w = img.shape[:2]
    m = img.copy()
    cv2.rectangle(m, (0, int(h*0.80)), (int(w*0.30), h), (0,0,0), -1)
    return m

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
    Returns (room, anomalies, None).
    """
    time.sleep(0.2)
    img = capture_screen()
    room = detect_room_name(img)
    tpl_img = template_images.get(room)
    if not room or tpl_img is None:
        return None, [], None

    img_m = mask_dynamic(img)
    tpl_m = mask_dynamic(tpl_img)

    anomalies = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for region in baseline_regions[room]:
        cls = region["class_name"]
        x1, y1, x2, y2 = map(int, region["box"])
        live_crop = img_m[y1:y2, x1:x2]
        tpl_crop  = tpl_m[y1:y2, x1:x2]

        diff = cv2.absdiff(live_crop, tpl_crop)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, binm = cv2.threshold(gray, BINARY_THRESH, 255, cv2.THRESH_BINARY)
        pix_count = int(np.count_nonzero(binm))

        if pix_count > PIXEL_COUNT_THRESHOLD:
            heat = cv2.applyColorMap(binm, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(live_crop, 0.7, heat, 0.5, 0)
            fname = f"{room}_{cls}_{ts}_HEAT.png"
            heat_path = os.path.join(BASE_DIR, room, fname)
            cv2.imwrite(heat_path, overlay)
            anomalies.append({
                "class_name": cls,
                "box": region["box"],
                "pixel_count": pix_count,
                "heatmap_path": heat_path
            })

    return room, anomalies, None
