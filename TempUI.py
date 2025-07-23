import os
import sys
import time
import keyboard
import json
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import backend  # your backend.py with process_room
import shutil
from backend import classify_heatmap, save_classification_signature
from PyQt5.QtMultimedia import QSoundEffect
# Constants
ROOMS            = backend.ROOMS
DEBOUNCE         = 0.5
BASE_DIR         = os.path.join(os.path.dirname(__file__), "LogCabin")
CLASS_MAP_PATH   = os.path.join(BASE_DIR, "classification_map.json")
ANOMALY_TYPES    = [
    "Dead body", "Door anomaly", "Extra object", "Image anomaly",
    "Intruder", "Missing object", "Object Manipulation",
    "Object Movement", "Object Replacement"
]
CLASSIFICATION_DIR = os.path.join(BASE_DIR, "classifications")

class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        flags = (QtCore.Qt.FramelessWindowHint |
                 QtCore.Qt.WindowStaysOnTopHint |
                 QtCore.Qt.Tool)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        screen = QtWidgets.QApplication.primaryScreen().size()
        self.setGeometry(0, 0, screen.width(), screen.height())
        self.anomalies = { r: [] for r in ROOMS }

        if os.path.exists(CLASS_MAP_PATH):
            with open(CLASS_MAP_PATH, "r") as f:
                self.class_map = json.load(f)
        else:
            self.class_map = {}

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)

        # Left column to hold all vertical panels
        left_column = QtWidgets.QFrame()
        left_column.setFixedWidth(300)
        left_column.setStyleSheet("background: rgba(255,255,255,230); border-radius:5px;")
        left_column_layout = QtWidgets.QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(5,5,5,5)
        left_column_layout.setSpacing(10)

        # Rooms box
        rooms_box = QtWidgets.QFrame()
        rooms_box.setStyleSheet("background:white; border:1px solid #AAA; border-radius:3px;")
        rooms_layout = QtWidgets.QVBoxLayout(rooms_box)
        rooms_layout.setContentsMargins(5,5,5,5)
        rooms_layout.setSpacing(5)

        self.left_labels  = {}
        self.left_buttons = {}

        for room in ROOMS:
            row = QtWidgets.QWidget()
            row_l = QtWidgets.QHBoxLayout(row)
            row_l.setContentsMargins(0,0,0,0)
            lbl = QtWidgets.QLabel(f"{room}: No data"); lbl.setFixedWidth(180)
            btn = QtWidgets.QPushButton("View"); btn.setEnabled(False)
            row_l.addWidget(lbl); row_l.addWidget(btn)
            rooms_layout.addWidget(row)
            self.left_labels[room]  = lbl
            self.left_buttons[room] = btn
            btn.clicked.connect(lambda _, r=room: self.openAnomalies(r))

        left_column_layout.addWidget(rooms_box, 0)

        # Center classification panel (dropdown + confirm button)
        self.center_panel = QtWidgets.QFrame()
        self.center_panel.setStyleSheet("background:white; border:1px solid #AAA; border-radius:3px;")
        self.center_layout = QtWidgets.QVBoxLayout(self.center_panel)
        self.center_layout.setContentsMargins(5,5,5,5)
        self.center_layout.setSpacing(5)

        self.class_display_label = QtWidgets.QLabel("Select classification type for current anomaly.")
        self.class_display_label.setWordWrap(True)
        self.center_layout.addWidget(self.class_display_label)

        self.class_dropdown = QtWidgets.QComboBox()
        self.class_dropdown.addItem("Select anomaly type…")
        self.class_dropdown.addItems(ANOMALY_TYPES)
        self.center_layout.addWidget(self.class_dropdown)

        self.confirm_button = QtWidgets.QPushButton("Confirm Classification")
        self.confirm_button.setEnabled(False)
        self.center_layout.addWidget(self.confirm_button)

        left_column_layout.addWidget(self.center_panel, 0)

        # Log box
        log_box = QtWidgets.QFrame()
        log_box.setStyleSheet("background:white; border:1px solid #AAA; border-radius:3px;")
        log_l = QtWidgets.QVBoxLayout(log_box)
        log_l.setContentsMargins(5,5,5,5)
        self.log_text = QtWidgets.QTextEdit(); self.log_text.setReadOnly(True)
        log_l.addWidget(self.log_text)
        left_column_layout.addWidget(log_box, 1)

        main_layout.addWidget(left_column, 0)

        # Right panel
        self.right_panel = QtWidgets.QFrame()
        self.right_panel.setStyleSheet("background:transparent;")
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0,0,0,0)
        self.right_layout.setSpacing(0)
        main_layout.addWidget(self.right_panel, 1)

        self.last6 = 0
        self.lastF12 = 0
        self.last8 = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.pollKeys)
        self.timer.start(50)

        # Anomaly sound
        self.anomaly_sound = QSoundEffect()
        self.anomaly_sound.setSource(QtCore.QUrl.fromLocalFile("anomaly.wav"))
        self.anomaly_sound.setVolume(0.8)  # 0.0 to 1.0

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0,0,0,160))
        super().paintEvent(ev)

    def toggle(self):
        self.setVisible(not self.isVisible())

    def appendLog(self, msg):
        self.log_text.append(f"{datetime.now().strftime('%H:%M:%S')} – {msg}")

    def clearRightPanel(self):
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
                child = item.layout()
                if child:
                    clear_layout(child)
        clear_layout(self.right_layout)

    def openAnomalies(self, room: str):
        
        anomalies = self.anomalies.get(room, [])
        lbl = self.left_labels[room]
        btn = self.left_buttons[room]
        lbl.setText(f"{room}: No data")
        lbl.setStyleSheet("color: black;")
        btn.setEnabled(False)
        self.clearRightPanel()

        self.class_display_label.setText(f"Room: {room}\nDetected anomalies: {len(anomalies)}")

        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setFixedSize(24,24)
        close_btn.setStyleSheet("background-color: rgba(255,255,255,200);")
        close_btn.clicked.connect(self.clearRightPanel)
        self.right_layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        self.grouped_heatmaps = []


        for a in anomalies:
            cls = a.get("class_name", a.get("class", "Unknown"))
            pix = a.get("pixel_count", "?")
            heat = a.get("heatmap_path", None)

            hdr = QtWidgets.QLabel(f"{cls}: {pix} px changed")
            hdr.setStyleSheet("color: white; font-weight: bold; margin: 0px; padding: 0px;")
            hdr.setContentsMargins(0, 0, 0, 0)
            self.right_layout.addWidget(hdr)

            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            container = QtWidgets.QWidget()
            container.setLayout(row)
            container.setContentsMargins(0, 0, 0, 0)
            self.right_layout.addWidget(container)

            if heat and os.path.exists(heat):
                self.grouped_heatmaps.append(heat)
                pm = QtGui.QPixmap(heat).scaled(400,400,QtCore.Qt.KeepAspectRatio)
                lbl_h = QtWidgets.QLabel(); lbl_h.setPixmap(pm)
                row.addWidget(lbl_h)
            else:
                row.addWidget(QtWidgets.QLabel("(no heatmap)"))

            tpl = os.path.join(BASE_DIR, room, "group_templates", f"{cls}.png")
            if os.path.exists(tpl):
                pm2 = QtGui.QPixmap(tpl).scaled(400,400,QtCore.Qt.KeepAspectRatio)
                lbl_t = QtWidgets.QLabel(); lbl_t.setPixmap(pm2)
                row.addWidget(lbl_t)
            else:
                row.addWidget(QtWidgets.QLabel("(no template)"))

        self.confirm_button.setEnabled(True)
        try:
            self.confirm_button.clicked.disconnect()
        except:
            pass
        self.confirm_button.clicked.connect(lambda: self.handle_classification(room, self.grouped_heatmaps))


    def pollKeys(self):
        now = time.time()
        if keyboard.is_pressed("f12") and now - self.lastF12 > DEBOUNCE:
            self.lastF12 = now
            self.toggle()
        if keyboard.is_pressed("6") and now - self.last6 > DEBOUNCE:
            self.last6 = now
            try:
                room, anom, _ = backend.process_room()
            except Exception as e:
                print("process_room error:", e)
                return
            if not room:
                print("[No room detected]")
            else:
                self.anomalies[room] = anom
                lbl = self.left_labels[room]; btn = self.left_buttons[room]
                if anom:
                    self.anomaly_sound.play()
                    lbl.setText(f"{room}: {len(anom)} anomaly(s)")
                    lbl.setStyleSheet("color:red;"); btn.setEnabled(True)
                    self.appendLog(
                        f"{room}: {len(anom)} anomalies – " + ", ".join(f"{a.get('class_name', a.get('class'))}({a.get('pixel_count', '?')})" for a in anom)
                    )
                else:
                    lbl.setText(f"{room}: No anomalies")
                    lbl.setStyleSheet("color:lightgreen;"); btn.setEnabled(False)
                    self.appendLog(f"{room}: No anomalies detected")
        if keyboard.is_pressed("8") and now - self.last8 > DEBOUNCE:
            self.last8 = now
            QtWidgets.QApplication.instance().quit()

    def handle_classification(self, room, heatmap_paths):
        atype = self.class_dropdown.currentText()
        if atype == "Select anomaly type…":
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a valid anomaly type.")
            return

        if not heatmap_paths:
            QtWidgets.QMessageBox.warning(self, "Warning", "No heatmaps selected.")
            return  

        try:
            saved_paths = []
            for path in heatmap_paths:
                saved_path = backend.classify_heatmap(room, path, atype)
                backend.save_classification_signature(path, atype)
                saved_paths.append(saved_path)

            QtWidgets.QMessageBox.information(self, "Saved", f"Saved {len(saved_paths)} heatmap(s) as '{atype}'.")
            self.appendLog(f"Saved classification: {atype} for {room} – {len(saved_paths)} heatmap(s)")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
