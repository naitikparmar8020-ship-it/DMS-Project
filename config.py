# config.py

# Threshold Targets (Partner A's math values)
EAR_THRESHOLD = 0.25      # Below 0.25 = Eyes closed
MAR_THRESHOLD = 0.45      # Above 0.65 = Mouth open wide (yawn)

# Consecutive Frame Requirements (~30 FPS camera feed)
EAR_CONSEC_FRAMES = 48    # ~1.5 - 2.0 seconds of closed eyes triggers alarm
MAR_CONSEC_FRAMES = 30    # ~1.0 second of continuous yawning triggers alert

# File Paths
ALARM_PATH = "assets/alarm.wav"

# UI Colors (OpenCV format: BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)