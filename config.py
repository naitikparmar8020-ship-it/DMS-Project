# Threshold Targets (Fallback values for Auto-Calibration)
EAR_THRESHOLD = 0.25      # Below 0.25 = Eyes closed
MAR_THRESHOLD = 0.60      # Above 0.65 = Mouth open wide (yawn)

# Phone Distraction
HAND_EAR_PROXIMITY_THRESHOLD = 250 # Increased for high-resolution distance

# Consecutive Frame Requirements (Tuned for low FPS)
DROWSINESS_CONSEC_FRAMES = 6  # ~1.5 seconds at current FPS
YAWN_CONSEC_FRAMES = 20      # ~1.5 seconds at current FPS
PHONE_CONSEC_FRAMES = 10       # ~1.5 seconds at current FPS
LOOKDOWN_CONSEC_FRAMES = 6

# File Paths
ALARM_PATH = "assets/alarm.wav"
PHONE_PATH = "assets/talking_on_phon.wav"

# UI Colors (OpenCV format: BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)