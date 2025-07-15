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
        btn_exit = QtWidgets.QPushButton("Exit", self)
        btn_exit.setGeometry(screen.width()-110, 20, 80, 30)
        btn_exit.setStyleSheet("background-color: rgba(255,255,255,200); font-weight:bold;")
        btn_exit.clicked.connect(QtWidgets.QApplication.instance().quit)

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
        self.right_layout.setContentsMargins(5,5,5,5)
        self.right_layout.setSpacing(10)
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

    def clearRight(self):
        while self.right_layout.count():
            w = self.right_layout.takeAt(0).widget()
            if w: w.deleteLater()

    def openAnomalies(self, room):
        # reset row
        lbl = self.left_labels[room]; btn = self.left_buttons[room]
        lbl.setText(f"{room}: No data"); lbl.setStyleSheet("color:black;")
        btn.setEnabled(False)

        self.clearRight()
        # close button
        c = QtWidgets.QPushButton("✕"); c.setFixedSize(24,24)
        c.setStyleSheet("background: rgba(255,255,255,200);")
        c.clicked.connect(self.clearRight)
        self.right_layout.addWidget(c, alignment=QtCore.Qt.AlignRight)

        for a in self.anomalies[room]:
            hdr = QtWidgets.QLabel(f"{a['class_name']}: {a['pixel_count']} px")
            hdr.setStyleSheet("color:white; font-weight:bold;")
            self.right_layout.addWidget(hdr)
            cont = QtWidgets.QHBoxLayout()
            self.right_layout.addLayout(cont)
            # heatmap
            hp = a['heatmap_path']
            if hp and os.path.exists(hp):
                pm = QtGui.QPixmap(hp).scaled(400,400,QtCore.Qt.KeepAspectRatio)
                l = QtWidgets.QLabel(); l.setPixmap(pm); cont.addWidget(l)
            else:
                cont.addWidget(QtWidgets.QLabel("(no heatmap)"))
            # template
            tp = os.path.join(BASE_DIR, room, "group_templates", f"{a['class_name']}.png")
            if os.path.exists(tp):
                pm = QtGui.QPixmap(tp).scaled(400,400,QtCore.Qt.KeepAspectRatio)
                l2 = QtWidgets.QLabel(); l2.setPixmap(pm); cont.addWidget(l2)
            else:
                cont.addWidget(QtWidgets.QLabel("(no tpl)"))

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
