import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# ───── CONFIG ──────────────────────────────────────────────────────
ROOMS    = ["Living", "Kitchen", "Bedroom", "Bathroom", "Entryway", "Yard"]
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")
# Maximum display size for the cropping window
MAX_W, MAX_H = 1920, 1080
# ────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for room in ROOMS:
        d = os.path.join(BASE_DIR, room, "group_templates")
        os.makedirs(d, exist_ok=True)

class CropTool:
    def __init__(self, room):
        self.room = room
        tpl_path = os.path.join(BASE_DIR, room, "template.png")
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"No template.png for {room}")

        # Load original image
        self.orig_img = Image.open(tpl_path)
        ow, oh = self.orig_img.size

        # Compute scale to fit within MAX_W x MAX_H
        scale = min(MAX_W / ow, MAX_H / oh, 1.0)
        self.scale = scale
        dw, dh = int(ow * scale), int(oh * scale)

        # Resize for display using LANCZOS (high-quality downsampling)
        disp_img = self.orig_img.resize((dw, dh), resample=Image.LANCZOS)

        # Create the Tk root before any PhotoImage
        self.root = tk.Tk()
        self.root.title(f"Crop: {room}")
        self.tkimg = ImageTk.PhotoImage(disp_img, master=self.root)

        # Canvas setup
        self.canvas = tk.Canvas(self.root, width=dw, height=dh, cursor="cross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        # Bind mouse events
        self.start = None
        self.rect  = None
        self.canvas.bind("<ButtonPress-1>",    self.on_button_press)
        self.canvas.bind("<B1-Motion>",        self.on_move)
        self.canvas.bind("<ButtonRelease-1>",  self.on_button_release)

        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", pady=5)
        tk.Button(btn_frame, text="Next Room", command=self.on_next).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Quit",      command=self.on_quit).pack(side="right", padx=5)

        self.root.mainloop()

    def on_button_press(self, event):
        self.start = (event.x, event.y)
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

    def on_move(self, event):
        if not self.start:
            return
        x0, y0 = self.start
        x1, y1 = event.x, event.y
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(x0, y0, x1, y1,
                                                 outline="red", width=2)

    def on_button_release(self, event):
        if not self.start:
            return
        x0, y0 = self.start
        x1, y1 = event.x, event.y
        x0, x1 = sorted((max(0, x0), min(self.tkimg.width(), x1)))
        y0, y1 = sorted((max(0, y0), min(self.tkimg.height(), y1)))
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            messagebox.showwarning("Too small", "Drag a larger area")
            return

        name = simpledialog.askstring("Class name", f"Name this crop in '{self.room}':")
        if not name:
            return

        # Map cropped coordinates back to original image
        ox0, oy0 = int(x0 / self.scale), int(y0 / self.scale)
        ox1, oy1 = int(x1 / self.scale), int(y1 / self.scale)
        crop = self.orig_img.crop((ox0, oy0, ox1, oy1))

        out_dir = os.path.join(BASE_DIR, self.room, "group_templates")
        out_path = os.path.join(out_dir, f"{name}.png")
        crop.save(out_path)
        messagebox.showinfo("Saved", f"Saved to:\n{out_path}")

        # Clear rectangle
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
        self.start = None

    def on_next(self):
        self.root.destroy()

    def on_quit(self):
        if messagebox.askyesno("Quit", "Abort cropping?"):
            self.root.destroy()
            raise SystemExit

if __name__ == "__main__":
    ensure_dirs()
    for room in ROOMS:
        try:
            CropTool(room)
        except FileNotFoundError:
            print(f"[SKIP] No template in {room}")
        except SystemExit:
            print("[ABORT] Cropping cancelled.")
            break
    print("✅ Cropping complete.")
