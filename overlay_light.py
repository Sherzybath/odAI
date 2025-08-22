
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
LOG_FILE = "automation.log"   # will be created in program’s working dir

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

        self.autohotkey_exe = r"C:\Program Files\AutoHotkey\v2\AutoHotkey.exe"
        self.ahk_script_path = r"C:\Users\sdzyr\Desktop\Project\press_right.ahk"
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
        line = f"[{timestamp}] {msg}"
        
        # --- write to UI ---
        self.log_panel.append(line)

        # --- write to file ---
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            # fail-safe: log errors only to console
            print(f"Logging error: {e}")
    

    def pause_detection(self):
        """Stop AHK (F8) and pause the detector thread while we resolve an anomaly."""
        # Stop AHK
        try:
            keyboard.press_and_release("f8")
            self.appendLog("AHK script paused via F8.")
        except Exception as e:
            self.appendLog(f"AHK pause (F8) failed: {e}")
        self.ahk_process = None  # we no longer track a running proc while paused

        # Stop detector thread
        if self.detector:
            self.detector.stop()
        if self.detector_thread and self.detector_thread.isRunning():
            self.detector_thread.quit()
            self.detector_thread.wait()
        self.detector = None
        self.detector_thread = None
        self.appendLog("Anomaly detection paused (thread stopped).")


    def resume_detection(self):
        time.sleep(1.7)
        """Restart AHK and resume the detector loop."""
        # Start AHK
        try:
            if self.autohotkey_exe and self.ahk_script_path:
                self.ahk_process = subprocess.Popen([self.autohotkey_exe, self.ahk_script_path])
                self.appendLog("AHK script restarted.")
        except Exception as e:
            self.appendLog(f"Failed to start AHK script: {e}")
        try:
            pyautogui.moveTo(5, 5, duration=0.2)  # small offset from (0,0) for safety
            self.appendLog("Mouse moved to top-left to reset position.")
        except Exception as e:
            self.appendLog(f"Could not move mouse to top-left: {e}")
        # Restart detector thread
        if self.detector is not None:
            self.appendLog("Detection already running.")
            return
        self.detector = AnomalyDetector()
        self.detector_thread = QtCore.QThread()
        self.detector.moveToThread(self.detector_thread)
        self.detector.resultReady.connect(self.handle_detection)
        self.detector.errorOccurred.connect(self.handle_error)
        self.detector_thread.started.connect(self.detector.run)
        self.detector_thread.start()
        self.appendLog("Anomaly detection resumed.")


    
    def start_automation(self):
        if self.automating:
            self.appendLog("Automation is already running.")
            return

        self.setVisible(False)
        self.visible = False
        self.automating = True
        self.appendLog("Automation started. Overlay hidden.")
        try:
            pyautogui.moveTo(5, 5, duration=0.2)  # small offset from (0,0) for safety
            self.appendLog("Mouse moved to top-left to reset position.")
        except Exception as e:
            self.appendLog(f"Could not move mouse to top-left: {e}")
        # Launch AHK using instance paths
        try:
            self.ahk_process = subprocess.Popen([self.autohotkey_exe, self.ahk_script_path])
            self.appendLog("AHK script started.")
        except Exception as e:
            self.appendLog(f"Failed to start AHK script: {e}")
            self.ahk_process = None

        # Start detector
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
        if not room:
            self.appendLog("Room detection failed.")
            return

        self.appendLog(f"Room: {room} – Detected {len(anomalies)} anomaly/anomalies.")
        if not anomalies:
            return

        # Pause everything (AHK via F8 + detector thread) while we resolve
        self.pause_detection()
        self.notify_anomaly(f"Anomaly detected in {room}!")

        try:
            coords = backend.get_anomaly_coordinates(anomalies)
        except Exception as e:
            self.appendLog(f"Error while selecting anomaly coordinates: {e}")
            # Even on error, resume automation
            self.resume_detection()
            return

        if not coords:
            self.appendLog("No anomaly coordinates found.")
            self.resume_detection()
            return

        # Resolve the anomaly (dropdown select + confirmation loop)
        try:
            winner = select_by_rank(
    coords=coords,
    heatmap_path=heat_path,
    logger=self.appendLog,
    max_distance=6,
    hold_seconds=2.0,
    step_px=60,
    pause=0.18,
    wait_seconds=5,
    room=room,   # <— important
)
            # backend.move_cursor_to_anomaly(coords, hold_seconds=2, dropdown=True, logger=self.appendLog)
        except Exception as e:
            self.appendLog(f"Error moving cursor to anomaly: {e}")

        # Small cushion (optional) before resuming
        time.sleep(0.5)

        # Resume AHK + detector for next room scan
        self.resume_detection()
    # def handle_detection(self, room, anomalies, heat_path):
    # # ensure variable is defined to avoid "referenced before assignment" error

    #     if not room:
    #         self.appendLog("Room detection failed.")
    #         return

    #     self.appendLog(f"Room: {room} – Detected {len(anomalies)} anomaly/anomalies.")
    #     if not anomalies:
    #         return
    #     self.stop_automation()

    #     self.notify_anomaly(f"Anomaly detected in {room}!")

    #     try:
    #         coords = backend.get_anomaly_coordinates(anomalies)
    #     except Exception as e:
    #         self.appendLog(f"Error while selecting anomaly coordinates: {e}")
    #         return

    #     if not coords:
    #         self.appendLog("No anomaly coordinates found.")
    #         return

    #     try:
    #         select_by_rank(
    # coords=coords,
    # heatmap_path=heat_path,   # pass the heatmap for the chosen anomaly
    # logger=self.appendLog,            # optional
    # max_distance=6,                   # tweak tolerance
    # hold_seconds=2.0,                 # your long-press
    # step_px=60,                       # your menu spacing
    # pause=0.18                        # hover time per option
    #     )
            
    #     except Exception as e:
    #         self.appendLog(f"Error moving cursor to anomaly: {e}")
            
        

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
