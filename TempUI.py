import os
import sys
import time
import keyboard
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
import backend  # your backend.py with process_room()

# Constants
ROOMS    = backend.ROOMS
DEBOUNCE = 0.5
BASE_DIR = os.path.join(os.path.dirname(__file__), "LogCabin")

class Overlay(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Frameless, always-on-top, tool window
        flags = (QtCore.Qt.FramelessWindowHint |
                 QtCore.Qt.WindowStaysOnTopHint |
                 QtCore.Qt.Tool)
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        screen = QtWidgets.QApplication.primaryScreen().size()
        self.setGeometry(0, 0, screen.width(), screen.height())

        # Layouts
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)

        # Exit button
        # btn_exit = QtWidgets.QPushButton("Exit", self)
        # btn_exit.setGeometry(screen.width()-110, 20, 80, 30)
        # btn_exit.setStyleSheet("background-color: rgba(255,255,255,200); font-weight:bold;")
        # btn_exit.clicked.connect(QtWidgets.QApplication.instance().quit)

        # Left panel
        left_panel = QtWidgets.QFrame()
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("background: rgba(255,255,255,230); border-radius:5px;")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5,5,5,5)
        left_layout.setSpacing(10)

        # Rooms box
        rooms_box = QtWidgets.QFrame()
        rooms_box.setStyleSheet("background:white; border:1px solid #AAA; border-radius:3px;")
        rooms_layout = QtWidgets.QVBoxLayout(rooms_box)
        rooms_layout.setContentsMargins(5,5,5,5)
        rooms_layout.setSpacing(5)

        self.left_labels  = {}
        self.left_buttons = {}
        self.anomalies    = {r: [] for r in ROOMS}

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

        left_layout.addWidget(rooms_box, 0)

        # Log box
        log_box = QtWidgets.QFrame()
        log_box.setStyleSheet("background:white; border:1px solid #AAA; border-radius:3px;")
        log_l = QtWidgets.QVBoxLayout(log_box)
        log_l.setContentsMargins(5,5,5,5)
        self.log_text = QtWidgets.QTextEdit(); self.log_text.setReadOnly(True)
        log_l.addWidget(self.log_text)
        left_layout.addWidget(log_box, 1)

        main_layout.addWidget(left_panel, 0)

        # Right panel
        self.right_panel = QtWidgets.QFrame()
        self.right_panel.setStyleSheet("background:transparent;")
        self.right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(2,2,2,2)
        self.right_layout.setSpacing(5)
        main_layout.addWidget(self.right_panel, 1)

        # Debounce state
        self.last6 = 0
        self.lastF12 = 0
        self.last8 = 0

        # Timer to poll keys on the main thread
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.pollKeys)
        self.timer.start(50)

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0,0,0,160))
        super().paintEvent(ev)

    def toggle(self):
        self.setVisible(not self.isVisible())

    def appendLog(self, msg):
        self.log_text.append(f"{datetime.now().strftime('%H:%M:%S')} – {msg}")

    def clearRightPanel(self):
        """Recursively remove every widget and layout from the right panel."""
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                # if it's a widget, delete it
                w = item.widget()
                if w:
                    w.deleteLater()
                # if it's a nested layout, clear that too
                child = item.layout()
                if child:
                    clear_layout(child)

        clear_layout(self.right_layout)

    def openAnomalies(self, room: str):
        anomalies = self.anomalies.get(room, [])

        # Reset that room’s row state
        lbl = self.left_labels[room]
        btn = self.left_buttons[room]
        lbl.setText(f"{room}: No data")
        lbl.setStyleSheet("color: black;")
        btn.setEnabled(False)

        # 1) Clear out anything left in right panel
        self.clearRightPanel()

        # 2) Add a close ("✕") button at the top
        close_btn = QtWidgets.QPushButton("✕")
        close_btn.setFixedSize(24,24)
        close_btn.setStyleSheet("background-color: rgba(255,255,255,200);")
        close_btn.clicked.connect(self.clearRightPanel)
        self.right_layout.addWidget(close_btn, alignment=QtCore.Qt.AlignRight)

        # 3) Repopulate with the new anomaly images
        for a in anomalies:
            cls = a["class_name"]
            pix = a["pixel_count"]
            heat_path = a["heatmap_path"]

            hdr = QtWidgets.QLabel(f"{cls}: {pix} px changed")
            hdr.setStyleSheet("color: white; font-weight: bold;")
            self.right_layout.addWidget(hdr)

            row = QtWidgets.QHBoxLayout()
            self.right_layout.addLayout(row)

            # Heatmap
            if heat_path and os.path.exists(heat_path):
                pixmap = QtGui.QPixmap(heat_path).scaled(
                    400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                lbl_h = QtWidgets.QLabel()
                lbl_h.setPixmap(pixmap)
                row.addWidget(lbl_h)
            else:
                row.addWidget(QtWidgets.QLabel("(no heatmap)"))

            # Template crop
            tpl_path = os.path.join(BASE_DIR, room, "group_templates", f"{cls}.png")
            if os.path.exists(tpl_path):
                pixmap = QtGui.QPixmap(tpl_path).scaled(
                    400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
                )
                lbl_t = QtWidgets.QLabel()
                lbl_t.setPixmap(pixmap)
                row.addWidget(lbl_t)
            else:
                row.addWidget(QtWidgets.QLabel("(no template)"))

    def pollKeys(self):
        now = time.time()
        # F12 toggle
        if keyboard.is_pressed("f12") and now - self.lastF12 > DEBOUNCE:
            self.lastF12 = now
            self.toggle()
        # check anomalies
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
                    lbl.setText(f"{room}: {len(anom)} anomaly(s)")
                    lbl.setStyleSheet("color:red;"); btn.setEnabled(True)
                    self.appendLog(
                        f"{room}: {len(anom)} anomalies – " +
                        ", ".join(f"{a['class_name']}({a['pixel_count']})" for a in anom)
                    )
                else:
                    lbl.setText(f"{room}: No anomalies")
                    lbl.setStyleSheet("color:lightgreen;"); btn.setEnabled(False)
                    self.appendLog(f"{room}: No anomalies detected")
        # Exit
        if keyboard.is_pressed("8") and now - self.last8 > DEBOUNCE:
            self.last8 = now
            QtWidgets.QApplication.instance().quit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()
    overlay.show()
    sys.exit(app.exec_())
