# build_classification_map.py
# Scan LogCabin/classifications/*/* images, compute dhash, and write LogCabin/classification_map.json

import os, json, argparse, sys
import cv2
from glob import glob

# ---- Defaults (match your codebase) -----------------------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classifications")
CLASSIFICATION_MAP = os.path.join(BASE_DIR, "classification_map.json")

# ---- dhash (same logic as selector.py) --------------------------------------
def dhash_signature(image_path: str) -> str | None:
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

# ---- IO helpers --------------------------------------------------------------
def load_map(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_map(path: str, mapping: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)

# ---- Main build --------------------------------------------------------------
def build_map(classifications_dir: str,
              out_json: str,
              overwrite_on_conflict: bool = False,
              exts: tuple = (".png", ".jpg", ".jpeg"),
              dry_run: bool = False) -> dict:
    """
    Walk <classifications_dir>/<room>/<label>/*.{png,jpg,jpeg}
    Map dhash -> label. If collisions occur:
      - overwrite_on_conflict=False: keep existing, print a warning
      - overwrite_on_conflict=True : replace with new label
    Returns the resulting map (not yet written if dry_run).
    """
    if not os.path.isdir(classifications_dir):
        print(f"[ERROR] No directory: {classifications_dir}")
        return {}

    mapping = load_map(out_json)
    added = updated = skipped = 0

    # pattern: LogCabin/classifications/*/*/*.(ext)
    for room_dir in sorted(glob(os.path.join(classifications_dir, "*"))):
        if not os.path.isdir(room_dir):
            continue
        for label_dir in sorted(glob(os.path.join(room_dir, "*"))):
            if not os.path.isdir(label_dir):
                continue
            label = os.path.basename(label_dir)  # anomaly type
            files = []
            for e in exts:
                files.extend(glob(os.path.join(label_dir, f"*{e}")))
            for fp in files:
                sig = dhash_signature(fp)
                if sig is None:
                    print(f"[WARN] Failed to read image: {fp}")
                    continue

                if sig in mapping:
                    if mapping[sig] == label:
                        skipped += 1
                    else:
                        msg = f"[CONFLICT] {sig} was '{mapping[sig]}', new='{label}' from {os.path.relpath(fp)}"
                        if overwrite_on_conflict:
                            print(msg + " -> overwriting")
                            mapping[sig] = label
                            updated += 1
                        else:
                            print(msg + " -> keeping existing")
                            skipped += 1
                else:
                    mapping[sig] = label
                    added += 1

    print(f"[SUMMARY] added={added} updated={updated} skipped={skipped} total={len(mapping)}")
    if not dry_run:
        save_map(out_json, mapping)
        print(f"[WRITE] {out_json}")
    else:
        print("[DRY RUN] No file written.")
    return mapping

# ---- CLI --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build classification_map.json from saved classifications.")
    ap.add_argument("--base", default=BASE_DIR, help="Path to LogCabin base dir (default: ./LogCabin)")
    ap.add_argument("--classifications", default=None, help="Path to classifications dir (default: <base>/classifications)")
    ap.add_argument("--out", default=None, help="Output JSON (default: <base>/classification_map.json)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite on hash->label conflicts")
    ap.add_argument("--dry-run", action="store_true", help="Compute but don't write JSON")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    cls_dir = os.path.abspath(args.classifications or os.path.join(base, "classifications"))
    out_json = os.path.abspath(args.out or os.path.join(base, "classification_map.json"))

    print(f"[INFO] base={base}")
    print(f"[INFO] scan={cls_dir}")
    print(f"[INFO] out ={out_json}")
    build_map(cls_dir, out_json, overwrite_on_conflict=args.overwrite, dry_run=args.dry_run)

if __name__ == "__main__":
    sys.exit(main())
