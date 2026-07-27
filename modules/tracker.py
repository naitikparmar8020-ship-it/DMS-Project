import cv2
import numpy as np
import pygame
import config


class DrowsinessTracker:
    """
        Module: State Logic, Temporal Tracking, and Audio Alerts.
        
        Tracks consecutive frames for low Eye Aspect Ratio (EAR), high Mouth 
        Aspect Ratio (MAR), and Hand-to-Ear proximity to detect drowsiness,
        yawning, and phone distraction in real-time.
       """
   
    def __init__(self):
        # 1. Temporal Frame Counters (persist state across video frames)
        self.eye_counter = 0
        self.mouth_counter = 0
        self.phone_counter = 0

        # 2. State Flags
        self.drowsy_alert = False
        self.yawn_alert = False
        self.phone_alert = False

        # 3. Audio Alarm Initialization
        pygame.mixer.init()
        try:
            self.alarm_sound = pygame.mixer.Sound(config.ALARM_PATH)
            self.warning_sound = pygame.mixer.Sound(config.PHONE_PATH)
        except Exception as e:
            print(f"[ERROR] Could not load alarm sound : {e}")
            self.alarm_sound = None
            self.warning_sound = None

        self.alarm_playing = None

    def _play_alarm(self, sound_obj):
        """Helper method to start looping the audio alarm in a background thread."""
        # FIX: Only execute play/stop logic if the requested sound is DIFFERENT from what is currently playing
        if self.alarm_playing != sound_obj:
            pygame.mixer.stop()  # Master kill switch for all audio
            if sound_obj is not None:
                sound_obj.play(-1)  # Start the new sound exactly once
            self.alarm_playing = sound_obj

    def _stop_alarm(self):
        """Helper method to stop the audio alarm if active."""
        # FIX: Only execute the stop command if something is actually playing
        if self.alarm_playing is not None:
            pygame.mixer.stop()
            self.alarm_playing = None

    def update(self, ear, mar, pitch, yaw, hand_distance, phone_detected):
        """
        Updates frame counters based on incoming EAR and MAR values, determines 
        system alerts, manages sound, and returns UI display outputs.

        Parameters:
            ear (float or None): Eye Aspect Ratio calculated by Partner A.
            mar (float or None): Mouth Aspect Ratio calculated by Partner A.
            pitch (float) : Head pitch angle.
            yaw (float) : Head yaw angle.
            hand_distance (float) : Distance in pixels between hand and cheek.

        Returns:
            status_text (str): Message to display on screen.
            status_color (tuple BGR): OpenCV color code (Blue, Green, Red).
        """
        # Default State: Safe / Alert
        status_text = "Status: Safe"
        status_color = config.COLOR_GREEN

        # -------------------------------------------------------------
        # 1. EYE ASPECT RATIO (DROWSINESS) LOGIC
        # -------------------------------------------------------------
        if ear is not None and ear < config.EAR_THRESHOLD:
            self.eye_counter += 1
            if self.eye_counter >= config.EAR_CONSEC_FRAMES:
                self.drowsy_alert = True
        else:
            self.eye_counter = 0
            self.drowsy_alert = False

        # -------------------------------------------------------------
        # 2. MOUTH ASPECT RATIO (YAWNING) LOGIC
        # -------------------------------------------------------------
        if mar is not None and mar > config.MAR_THRESHOLD:
            self.mouth_counter += 1
            if self.mouth_counter >= config.MAR_CONSEC_FRAMES:
                self.yawn_alert = True
        else:
            self.mouth_counter = 0
            self.yawn_alert = False

        # -------------------------------------------------------------
        # 3. HAND-TO-EAR PROXIMITY (PHONE DISTRACTION) LOGIC
        # -------------------------------------------------------------
        # Get threshold from config or default to 20 frames
        phone_consec_frames = getattr(config, 'PHONE_CONSEC_FRAMES', 20)
        # Trigger if the phone seen
        if phone_detected or hand_distance < 80:
            self.phone_counter += 1
            if self.phone_counter >= phone_consec_frames:
                self.phone_alert = True
        else:
            self.phone_counter = 0
            self.phone_alert = False
        
        if hand_distance < 50:
            self.phone_counter += 1
            if self.phone_counter >= phone_consec_frames:
                self.phone_alert = True
        else:
            self.phone_counter = 0
            self.phone_alert = False

        # -------------------------------------------------------------
        # 4. PRIORITY STATE RESOLUTION & ALERTS
        # -------------------------------------------------------------
        if self.drowsy_alert:
            status_text = "WARNING: DROWSY!"
            status_color = config.COLOR_RED
            self._play_alarm(self.alarm_sound)

        elif self.phone_alert:
            status_text = "WARNING: PHONE DISTRACTION!"
            status_color = config.COLOR_RED
            self._play_alarm(self.warning_sound)    

        elif self.yawn_alert:
            status_text = "WARNING: YAWNING DETECTED!"
            status_color = config.COLOR_YELLOW
            self._stop_alarm()

        else:
            self._stop_alarm()

        return status_text, status_color

    def draw_ui(self, frame, ear, mar, status_text, status_color, left_eye_pts=None, right_eye_pts=None, mouth_pts=None, hand_pts=None, phone_box=None):
        """"
        Task 2 Implementation: Renders HUD elements, metrics, dynamic status,
        and facial landmark contours directly on the OpenCV frame.
        """ 
        # --- A. DRAW FACIAL OUTLINES ('cv2.polylines') ---
        # Draw green outlines around eyes and mouth if coordinates are passed by Partner A
        if left_eye_pts is not None and len(left_eye_pts) > 0:
            cv2.polylines(frame, [np.array(left_eye_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

        if right_eye_pts is not None and len(right_eye_pts) > 0:
             cv2.polylines(frame, [np.array(right_eye_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

        if mouth_pts is not None and len(mouth_pts) > 0:
            cv2.polylines(frame, [np.array(mouth_pts, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)

        if hand_pts is not None and len(hand_pts) > 0:
            for pt in hand_pts:
                cv2.circle(frame, pt, 4, (0, 165, 255), -1)
        
                 # Orange circles for hand landmarks
        if phone_box is not None:
            x1, y1, x2, y2 = phone_box
            # Rectangle around phone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # Add label
            cv2.putText(frame, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
             
        # --- C. DISPLAY REAL-TIME METRICS ('CV2.putText')
        ear_str = f"EAR: {ear:.2f}" if ear is not None else "EAR: N/A"
        mar_str = f"MAR: {mar:.2f}" if mar is not None else "MAR: N/A"

        # Top-left telemetry readings
        cv2.putText(frame, ear_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, mar_str, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) 
        cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)
        
        # --- D. DYNAMIC STATUS BANNER ---
        # Draw dynamic text with dynamic background color at top-center of the screen
        cv2.putText(frame, status_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

        return frame