# fast_intruder_to_top_easyocr_debug.py
# Find "Intruder" with EasyOCR, move cursor ~210px above, and save debug overlay

import time
import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab

# --- Config ---
DELAY_SECONDS   = 2.0
DELTA_TO_TOP    = 290
NUDGE_PX        = 12
SEARCH_PAD_XY   = (380, 260)
UPSCALE         = 1.45
DEBUG_SAVE      = "intruder_debug.png"

def _get_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=False)

def _prep_fast(gray):
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return norm, bw

def _is_intruder(txt: str) -> bool:
    if not txt: return False
    t = "".join(ch.lower() for ch in txt if ch.isalnum())
    return "intrud" in t

def _find_intruder_center(reader, img_gray):
    if UPSCALE > 1.0:
        img_gray = cv2.resize(img_gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)

    norm, bw = _prep_fast(img_gray)
    inv = 255 - bw

    def ocr_once(arr):
        return reader.readtext(arr, detail=1, paragraph=False, decoder="greedy")

    for variant in (norm, inv):
        results = ocr_once(variant)
        best = None
        for bbox, text, conf in results:
            if _is_intruder(text):
                xs = [int(p[0]) for p in bbox]
                ys = [int(p[1]) for p in bbox]
                cx = (min(xs) + max(xs)) // 2
                cy = (min(ys) + max(ys)) // 2
                if best is None or conf > best[0]:
                    best = (conf, cx, cy, bbox)
        if best:
            inv_scale = 1.0 / UPSCALE
            return int(best[1] * inv_scale), int(best[2] * inv_scale), best[3]
    return None

def main():
    reader = _get_reader()
    print(f"[INFO] Waiting {DELAY_SECONDS:.1f}s — hover mouse near dropdown.")
    time.sleep(DELAY_SECONDS)

    sw, sh = pyautogui.size()
    cx, cy = pyautogui.position()
    pad_x, pad_y = SEARCH_PAD_XY

    left, top   = max(0, cx - pad_x), max(0, cy - pad_y)
    right, bottom = min(sw, cx + pad_x), min(sh, cy + pad_y)

    roi_rgb = np.array(ImageGrab.grab(bbox=(left, top, right, bottom)))
    roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    res = _find_intruder_center(reader, roi_gray)
    if not res:
        print("[WARN] Could not find 'Intruder'.")
        return

    word_cx_local, word_cy_local, bbox = res
    word_cx_screen = left + word_cx_local
    word_cy_screen = top + word_cy_local

    anchor_x = np.clip(word_cx_screen, 0, sw - 1)
    anchor_y = np.clip(word_cy_screen - DELTA_TO_TOP + NUDGE_PX, 0, sh - 1)

    pyautogui.moveTo(anchor_x, anchor_y, duration=0.08)
    print(f"[RESULT] Moved to top row ({anchor_x}, {anchor_y}) from 'Intruder' at ({word_cx_screen}, {word_cy_screen}).")

    # --- Debug overlay ---
    dbg = roi_rgb.copy()
    # Draw bbox in red
    pts = np.array(bbox, dtype=np.int32)
    cv2.polylines(dbg, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    # Draw center dot in green
    cv2.circle(dbg, (word_cx_local, word_cy_local), 6, (0, 255, 0), -1)
    cv2.imwrite(DEBUG_SAVE, dbg[:, :, ::-1])  # RGB -> BGR for cv2.imwrite
    print(f"[DEBUG] Saved overlay to {DEBUG_SAVE}")

if __name__ == "__main__":
    main()
