
import sys
import time
import keyboard
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess
import psutil
import os
import backend
import pyautogui
from backend import move_cursor_to_anomaly
import numpy as np
pyautogui.FAILSAFE = False
from selector import select_by_rank

class AnomalyNotification(QtWidgets.QWidget):
    def __init__(self, message):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.Tool | QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setStyleSheet("""
            background-color: rgba(255, 0, 0, 220);
            color: white;
            font-weight: bold;
            padding: 10px;
            border-radius: 8px;
        """)
        layout = QtWidgets.QVBoxLayout()
        label = QtWidgets.QLabel(message)
        layout.addWidget(label)
        self.setLayout(layout)
        self.adjustSize()
        self.move(50, 50)
        QtCore.QTimer.singleShot(4000, self.close)


class AnomalyDetector(QtCore.QObject):
    resultReady = QtCore.pyqtSignal(str, list, str)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.detecting = True

    def stop(self):
        self.detecting = False

    def run(self):
        while self.detecting:
            try:
                room, anomalies, heat_path = backend.process_room()
                self.resultReady.emit(room, anomalies, heat_path)
            except Exception as e:
                self.errorOccurred.emit(str(e))
            time.sleep(5)

class OverlayLight(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        flags = (
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        screen = QtWidgets.QApplication.primaryScreen().size()
        self.setGeometry(0, 0, screen.width(), screen.height())

        self.visible = False
        self.setVisible(self.visible)
        self.automating = False
        self.ahk_process = None

        self.detector_thread = None
        self.detector = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        left_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(left_panel, 1)

        self.log_panel = QtWidgets.QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setStyleSheet("background-color: white; color: black; border: 1px solid gray;")
        left_panel.addWidget(self.log_panel)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start_automation)
        left_panel.addWidget(self.start_button)

        self.heatmap_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(self.heatmap_panel, 1)

        self.original_label = QtWidgets.QLabel("Original")
        self.heatmap_label = QtWidgets.QLabel("Heatmap")
        self.original_image = QtWidgets.QLabel()
        self.heatmap_image = QtWidgets.QLabel()
        self.original_image.setAlignment(QtCore.Qt.AlignCenter)
        self.heatmap_image.setAlignment(QtCore.Qt.AlignCenter)

        self.heatmap_panel.addWidget(self.original_label)
        self.heatmap_panel.addWidget(self.original_image)
        self.heatmap_panel.addWidget(self.heatmap_label)
        self.heatmap_panel.addWidget(self.heatmap_image)

        self.lastF12 = 0
        self.last8 = 0
        self.DEBOUNCE = 0.5

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.pollKeys)
        self.timer.start(50)
    
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 160))
        super().paintEvent(ev)

    def notify_anomaly(self, msg):
        self.alert = AnomalyNotification(msg)
        self.alert.show()

    def pollKeys(self):
        now = time.time()

        if keyboard.is_pressed("f12") and now - self.lastF12 > self.DEBOUNCE:
            self.lastF12 = now
            self.visible = not self.visible
            self.setVisible(self.visible)
            self.appendLog(f"Overlay toggled to {'visible' if self.visible else 'hidden'} with F12.")
            if self.automating:
                self.stop_automation()
                self.appendLog("Automation paused. AHK script and anomaly scan stopped.")

        if keyboard.is_pressed("8") and now - self.last8 > self.DEBOUNCE:
            self.last8 = now
            self.stop_automation()
            QtWidgets.QApplication.instance().quit()

    def appendLog(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.log_panel.append(f"[{timestamp}] {msg}")
    

    


    
    def start_automation(self):
        if self.automating:
            self.appendLog("Automation is already running.")
            return

        self.setVisible(False)
        self.visible = False
        self.automating = True
        self.appendLog("Automation started. Overlay hidden.")

        ahk_script_path = r"C:\Users\sdzyr\Desktop\Project\press_right.ahk"
        autohotkey_exe = r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe"
        self.ahk_process = subprocess.Popen([autohotkey_exe, ahk_script_path])
        self.appendLog("AHK script started.")

        self.detector = AnomalyDetector()
        self.detector_thread = QtCore.QThread()
        self.detector.moveToThread(self.detector_thread)
        self.detector.resultReady.connect(self.handle_detection)
        self.detector.errorOccurred.connect(self.handle_error)
        self.detector_thread.started.connect(self.detector.run)
        self.detector_thread.start()

    def stop_automation(self):
        if self.ahk_process:
            keyboard.press_and_release("f8")       
            self.ahk_process = None
            self.appendLog("AHK script stopped.")

        if self.detector:
            self.detector.stop()
            self.detector_thread.quit()
            self.detector_thread.wait()
            self.detector = None
            self.appendLog("Anomaly detection stopped.")

        self.automating = False

   

    def handle_detection(self, room, anomalies, heat_path):
    # ensure variable is defined to avoid "referenced before assignment" error
        heatmap_path = heat_path if heat_path else None

        if not room:
            self.appendLog("Room detection failed.")
            return

        self.appendLog(f"Room: {room} – Detected {len(anomalies)} anomaly/anomalies.")
        if not anomalies:
            return
        self.stop_automation()

        self.notify_anomaly(f"Anomaly detected in {room}!")

        try:
            coords = backend.get_anomaly_coordinates(anomalies)
        except Exception as e:
            self.appendLog(f"Error while selecting anomaly coordinates: {e}")
            return

        if not coords:
            self.appendLog("No anomaly coordinates found.")
            return

        try:
    #         select_by_rank(
    # coords=coords,
    # heatmap_path=heat_path,   # pass the heatmap for the chosen anomaly
    # logger=self.appendLog,            # optional
    # max_distance=6,                   # tweak tolerance
    # hold_seconds=2.0,                 # your long-press
    # step_px=60,                       # your menu spacing
    # pause=0.18                        # hover time per option
    #     )
            backend.move_cursor_to_anomaly(coords, hold_seconds=2, dropdown=True, logger=self.appendLog)
        except Exception as e:
            self.appendLog(f"Error moving cursor to anomaly: {e}")
            
        

    def handle_error(self, error_msg):
        self.appendLog(f"Error during detection: {error_msg}")

    def display_heatmap(self, heat_path):
        if not heat_path:
            self.appendLog("Heatmap path is None.")
            self.heatmap_image.setText("(heatmap not found)")
            self.original_image.setText("(original not found)") 
            return

        try:
            if os.path.exists(heat_path):
                heatmap = QtGui.QPixmap(heat_path).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
                self.heatmap_image.setPixmap(heatmap)
                self.appendLog("Passed test 1 – Heatmap displayed.")
            else:
                self.heatmap_image.setText("(heatmap not found)")
                self.appendLog("Heatmap not found on disk.")
        except Exception as e:
            self.appendLog(f"Error displaying heatmap: {e}")

        try:
            original_path = heat_path.replace("_HEAT", "")
            if os.path.exists(original_path):
                original = QtGui.QPixmap(original_path).scaled(400, 400, QtCore.Qt.KeepAspectRatio)
                self.original_image.setPixmap(original)
                self.appendLog("Passed test 2 – Original image displayed.")
            else:
                self.original_image.setText("(original not found)")
                self.appendLog("Original image not found on disk.")
        except Exception as e:
            self.appendLog(f"Error displaying original image: {e}")



def launch():
    app = QtWidgets.QApplication(sys.argv)
    overlay = OverlayLight()
    overlay.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
