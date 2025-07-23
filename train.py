import os
import sys
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk

# ───── CONFIG ──────────────────────────────────────────────────────
ROOMS    = ["Yard","Entryway","Living", "Kitchen", "Bedroom", "Bathroom"  ]
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")
MAX_W, MAX_H = 1920, 1080
# ────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for room in ROOMS:
        os.makedirs(os.path.join(BASE_DIR, room, "group_templates"), exist_ok=True)

class RectCropTool:
    def __init__(self, room):
        self.room = room
        tpl_path = os.path.join(BASE_DIR, room, "template.png")
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"No template.png for {room}")

        # Load original image
        self.orig_img = Image.open(tpl_path)
        ow, oh = self.orig_img.size

        # Compute scale to fit within MAX_W x MAX_H
        self.scale = min(MAX_W/ow, MAX_H/oh, 1.0)
        dw, dh = int(ow * self.scale), int(oh * self.scale)
        disp_img = self.orig_img.resize((dw, dh), Image.LANCZOS)

        # Set up window
        self.root = tk.Tk()
        self.root.title(f"Crop: {room}")
        self.tkimg = ImageTk.PhotoImage(disp_img, master=self.root)
        self.canvas = tk.Canvas(self.root, width=dw, height=dh, cursor="tcross")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)

        # State: two clicks
        self.points = []  # [(x0,y0), (x1,y1)]
        self.rect_id = None

        # Bind clicks
        self.canvas.bind("<Button-1>", self.on_click)
        self.root.bind("<Escape>", lambda e: self.on_skip())

        # Instructions
        instr = "Click once for top-left, click again for bottom-right.\nPress Esc to skip room."
        tk.Label(self.root, text=instr, bg="white").place(x=10, y=10)

        self.root.mainloop()

    def on_click(self, event):
        x, y = event.x, event.y
        if len(self.points) == 0:
            # First click
            self.points = [(x, y)]
            r = 4
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="red", outline="")
        else:
            # Second click
            self.points.append((x, y))
            x0, y0 = self.points[0]
            x1, y1 = self.points[1]
            # Normalize
            x0, x1 = sorted((max(0, x0), min(self.tkimg.width(), x1)))
            y0, y1 = sorted((max(0, y0), min(self.tkimg.height(), y1)))
            # Draw rectangle
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="blue", width=2)
            self.finish_crop()

    def finish_crop(self):
        x0, y0 = self.points[0]
        x1, y1 = self.points[1]
        # Minimum size check
        if abs(x1 - x0) < 10 or abs(y1 - y0) < 10:
            messagebox.showwarning("Too small", "Selection must be at least 10×10 pixels.")
            self.points = []
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            return

        # Ask for class name
        name = simpledialog.askstring("Class name", f"Name this crop in '{self.room}':")
        if not name:
            self.points = []
            return

        # Map back to original coordinates
        ox0, oy0 = int(x0 / self.scale), int(y0 / self.scale)
        ox1, oy1 = int(x1 / self.scale), int(y1 / self.scale)
        crop = self.orig_img.crop((ox0, oy0, ox1, oy1))

        # Save crop
        out_dir = os.path.join(BASE_DIR, self.room, "group_templates")
        out_path = os.path.join(out_dir, f"{name}.png")
        crop.save(out_path)
        messagebox.showinfo("Saved", f"Saved to:\n{out_path}")
        self.root.destroy()

    def on_skip(self):
        # Skip this room without saving
        self.root.destroy()

if __name__ == "__main__":
    ensure_dirs()
    for room in ROOMS:
        try:
            RectCropTool(room)
        except FileNotFoundError:
            print(f"[SKIP] No template for {room}")
    print("✅ Cropping complete.")

