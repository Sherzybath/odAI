import cv2
import numpy as np
import os

def clamp_box(box, img_shape):
    """Ensure the bounding box fits within the image dimensions"""
    height, width = img_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))
    if x1 >= x2 or y1 >= y2:
        return None  # Invalid box after clamping
    return [x1, y1, x2, y2]

def process_heatmap_debug(heatmap_path, box, output_dir="debug_heatmaps"):
    os.makedirs(output_dir, exist_ok=True)

    try:
        img = cv2.imread(heatmap_path)
        if img is None:
            print(f"[ERROR] Failed to read image: {heatmap_path}")
            return

        clamped_box = clamp_box(box, img.shape)
        if clamped_box is None:
            print(f"[ERROR] Invalid clamped box for: {heatmap_path}")
            return

        x1, y1, x2, y2 = clamped_box
        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            print(f"[ERROR] Empty crop from: {heatmap_path}")
            return

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        thresh_val = min(200, int(mean_val * 1.2))
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

        bright_pixels = cv2.countNonZero(binary)
        total_pixels = binary.size
        density = bright_pixels / total_pixels if total_pixels else 0

        # Compute centroid
        coords = cv2.findNonZero(binary)
        centroid = None
        if coords is not None:
            moments = cv2.moments(binary)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                centroid = (cx, cy)
                cv2.circle(binary, centroid, 10, (128), -1)

        base_name = os.path.basename(heatmap_path).replace(".png", "")
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_gray.png"), gray)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_bw.png"), binary)

        if centroid:
            print(f"[INFO] Processed {base_name}, Bright: {bright_pixels}, Density: {density:.4f}, Centroid: {centroid}")
        else:
            print(f"[INFO] Processed {base_name}, Bright: {bright_pixels}, Density: {density:.4f}, No centroid")

    except Exception as e:
        print(f"[EXCEPTION] {e}")

# Example test
if __name__ == "__main__":
    test_path = r"C:\Users\sdzyr\Desktop\Project\odAI\LogCabin\Bathroom\heatmaps\Bathroom_Sink_20250807_184517_HEAT.png"
    test_box = [633.0, 1.0, 1117.0, 697.0]  # Will be clamped safely
    process_heatmap_debug(test_path, test_box)