import cv2
import numpy as np
import os

def extract_red_regions_and_centroid(image_path, output_folder="bw_centroid_output"):
    os.makedirs(output_folder, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    # Convert to HSV to isolate red/orange heatmap areas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define two red ranges in HSV
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 120, 100])
    upper_red2 = np.array([180, 255, 255])

    # Mask for red/orange regions
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Invert for black/white visualization (red = black)
    bw_image = cv2.bitwise_not(red_mask)

    # Get coordinates of the black pixels (which were red in original)
    coords = np.column_stack(np.where(red_mask > 0))

    centroid = None
    density = 0.0
    if coords.size > 0:
        centroid_y, centroid_x = np.mean(coords, axis=0).astype(int)
        centroid = (centroid_x, centroid_y)
        total_pixels = img.shape[0] * img.shape[1]
        bright_pixels = coords.shape[0]
        density = bright_pixels / total_pixels
        cv2.circle(bw_image, centroid, radius=6, color=127, thickness=-1)

    # Save output
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, f"bw_centroid_{filename}")
    cv2.imwrite(output_path, bw_image)

    print(f"[✓] Processed: {filename}")
    print(f"    ➤ Centroid: {centroid}")
    print(f"    ➤ Density: {density:.4f}")
    print(f"    ➤ Saved at: {output_path}")
    print()

    return {
        "filename": filename,
        "centroid": centroid,
        "density": density,
        "output_path": output_path
    }

def process_folder(folder_path, output_folder="bw_centroid_output"):
    if not os.path.isdir(folder_path):
        print(f"[ERROR] Folder not found: {folder_path}")
        return

    heatmap_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"[INFO] Found {len(heatmap_files)} heatmaps.")
    for file_path in heatmap_files:
        extract_red_regions_and_centroid(file_path, output_folder)

# =======================
# 👇 Modify This Section
# =======================
if __name__ == "__main__":
    heatmap_folder = r"c:\Users\sdzyr\Desktop\Project\odAI\LogCabin\Bathroom\heatmaps"  # your folder
    output_folder = r"c:\Users\sdzyr\Desktop\Project\odAI\LogCabin\Bathroom\bw_centroid_output"
    process_folder(heatmap_folder, output_folder)