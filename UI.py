import os
import time
import keyboard
import tkinter as tk
from tkinter import Toplevel, Label, Scrollbar, Canvas, Frame
from PIL import Image, ImageTk
from datetime import datetime
import backend  # ensure this points to your updated backend.py

ROOMS = backend.ROOMS
DEBOUNCE = 0.5
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")

root = tk.Tk()
root.title("Observation Duty")

# ─── Left column ────────────────────────────────────────────────────────────

left_frame = tk.Frame(root)
left_frame.pack(side="left", fill="y", padx=10, pady=10)

left_vars = {}    # room -> (StringVar, Label)
left_buttons = {} # room -> Button

for room in ROOMS:
    row = tk.Frame(left_frame)
    row.pack(fill="x", pady=2)
    var = tk.StringVar(value=f"{room}: No data")
    lbl = tk.Label(row, textvariable=var, width=25, anchor="w", font=("Arial", 10))
    lbl.pack(side="left")
    btn = tk.Button(row, text="View", state="disabled", width=8)
    btn.pack(side="left", padx=5)
    left_vars[room] = (var, lbl)
    left_buttons[room] = btn

# ─── Right column (log) ─────────────────────────────────────────────────────

right_frame = tk.Frame(root)
right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

log_text = tk.Text(right_frame, state="disabled", width=50, height=20, wrap="none")
log_text.pack(side="left", fill="both", expand=True)
scrollbar = Scrollbar(right_frame, command=log_text.yview)
scrollbar.pack(side="right", fill="y")
log_text.configure(yscrollcommand=scrollbar.set)

def append_log(msg: str):
    log_text.config(state="normal")
    log_text.insert("end", f"{datetime.now().strftime('%H:%M:%S')} – {msg}\n")
    log_text.see("end")
    log_text.config(state="disabled")

def open_anomalies(room: str, anomalies: list):
    # reset UI state
    var, lbl = left_vars[room]
    var.set(f"{room}: No data")
    lbl.config(fg="black")
    left_buttons[room].config(state="disabled")

    # new pop-up window
    win = Toplevel(root)
    win.title(f"Anomalies in {room}")
    win.geometry("1200x900")
    win.minsize(800, 600)

    canvas = Canvas(win)
    scrollbar = Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = Frame(canvas)
    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0,0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for a in anomalies:
        cls = a.get("class_name", "unknown")
        pix = a.get("pixel_count", 0)
        heat_path = a.get("heatmap_path")

        # Header
        Label(scroll_frame,
              text=f"{cls}: {pix} px changed",
              font=("Arial", 12, "bold")
        ).pack(pady=(15,5))

        # Container for side-by-side images
        container = Frame(scroll_frame)
        container.pack(fill="x", pady=(0,15))

        # Left: heatmap
        if heat_path and os.path.exists(heat_path):
            img_h = Image.open(heat_path)
            img_h.thumbnail((550,550))
            photo_h = ImageTk.PhotoImage(img_h)
            lbl_h = Label(container, image=photo_h)
            lbl_h.image = photo_h
            lbl_h.pack(side="left", padx=5)
        else:
            Label(container, text="(no heatmap)", font=("Arial", 10)).pack(side="left", padx=5)

        # Right: original template crop
        tpl_path = os.path.join(BASE_DIR, room, "group_templates", f"{cls}.png")
        if os.path.exists(tpl_path):
            img_t = Image.open(tpl_path)
            img_t.thumbnail((550,550))
            photo_t = ImageTk.PhotoImage(img_t)
            lbl_t = Label(container, image=photo_t)
            lbl_t.image = photo_t
            lbl_t.pack(side="right", padx=5)
        else:
            Label(container, text="(no template)", font=("Arial", 10)).pack(side="right", padx=5)

def poll_keys():
    if keyboard.is_pressed("6"):
        room, anomalies, _ = backend.process_room()
        if room:
            var, lbl = left_vars[room]
            btn = left_buttons[room]
            if anomalies:
                count = len(anomalies)
                var.set(f"{room}: {count} anomaly(s)")
                lbl.config(fg="red")
                btn.config(
                    state="normal",
                    command=lambda r=room, a=anomalies: open_anomalies(r, a)
                )
                append_log(
                    f"{room}: {count} anomalies – " +
                    ", ".join(f"{a.get('class_name')}({a.get('pixel_count')})" for a in anomalies)
                )
            else:
                var.set(f"{room}: No anomalies")
                lbl.config(fg="green")
                append_log(f"{room}: No anomalies detected")
        time.sleep(DEBOUNCE)

    elif keyboard.is_pressed("8"):
        root.destroy()
        return

    root.after(100, poll_keys)

root.after(100, poll_keys)
root.mainloop()
