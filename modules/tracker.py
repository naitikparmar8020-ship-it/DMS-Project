import cv2
import numpy as np
import pygame
import config
import csv
import os
from datetime import datetime



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
        # self.phone_counter = 0
        self.yawn_counter = 0
        self.phone_history=[]
        self.history_length=30 #it look at last 10 frames
        self.alert_threshold = 15 #sound play if phone detected in those 10 frames 

        # 2. State Flags
        self.drowsy_alert = False
        self.yawn_alert = False
        self.phone_alert = False

        # dynamic calibration variables
        self.is_calibrating = True
        self.calib_frames_needed = 100 # captures roughly 3-5 seconds of video
        self.calib_frames_done = 0
        self.ear_sum = 0.0
        self.mar_sum = 0.0
        self.baseline_mar = 0.25 # fallback defaults
        self.baseline_mar = 0.60

        # CSV SESSION LOGGING VARIABLES
        self.log_file = "driver_log.csv"
        # if the file does not exist, create it and write the header
        if not os.path.exists(self.log_file):
                               
            with open (self.log_file , mode = 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event_Type", "Value"])

        # "LOCKS" to prevent spamming the CSV with 30 lines per second
        self.log_lock_drowsy = False
        self.log_lock_yawn = False
        self.log_lock_phone = False
         


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
    def log_event(self , event_name , value):
        with open(self.log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_name , f"{value:.2f}"])

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
        # ==========================================
        #  DYNAMIC AUTO-CALIBRATION
        # ==========================================
        if self.is_calibrating:
            if ear is not None and mar is not None:
                self.ear_sum += ear
                self.mar_sum += mar
                self.calib_frames_done += 1
                
                # Once we collect 100 frames, calculate the averages and unlock the system!
                if self.calib_frames_done >= self.calib_frames_needed:
                    self.baseline_ear = self.ear_sum / self.calib_frames_needed
                    self.baseline_mar = self.mar_sum / self.calib_frames_needed
                    self.is_calibrating = False
                    self.log_event("Session_Started", 0)
                    print(f"Calibration Complete! Baseline EAR: {self.baseline_ear:.2f} | MAR: {self.baseline_mar:.2f}")
                    
            return f"CALIBRATING ({self.calib_frames_done}/{self.calib_frames_needed})", (0, 255, 255)
        # we have to new line of code here 
        # ==========================================
        # PHASE 2: EVENT DETECTION & LOGGING
        # ==========================================
        #  calculate dynamic thresolds for every driver
        drowsy_thres=self.baseline_ear * 0.75
        yawn_thres=self.baseline_mar * 1.2
        phone_consec_frames = getattr(config , 'PHONE_CONSEC_FRAMES', 7)

        # drowsiness EAR
        if ear < drowsy_thres:
            self.drowsy_counter += 1
            if self.drowsy_counter >=getattr(config , 'DROWSINESS_CONSEC_FRAMES', 20):
                self.drowsy_alert = True
                # write to CSV once per incidence
                self.log_event("Drowsiness Detected",mar)
                self.log_lock_drowsy =True
        else:
            self.drowsy_counter =0
            self.drowsy_alert = False
            self.log_lock_drowsy =False
        #  yawning MAR
        if mar > yawn_thres:
            self.yawn_counter +=1
            if self.yawn_counter >= getattr(config , 'YAWN_CONSEC_FRAME', 15):
                self.yawn_alert = True
                if not self.log_lock_yawn:
                    self.log_event("Yawn Detected" ,mar)
                    self.log_lock_yawn = True
        else:
            self.yawn_counter = 0
            self.yawn_alert = False
            self.log_lock_yawn =False
        # phone distraction
        current_detection=(phone_detected and hand_distance < 250)
        self.phone_history.append(current_detection)
        if len(self.phone_history) > self.history_length:
            self.phone_history.pop(0)
        if sum(self.phone_history)>=self.alert_threshold:
                self.phone_alert=True
                if not self.log_lock_phone:
                    self.log_event("Phone Distraction",hand_distance)
                    self.log_lock_phone=True
        else:
            # self.phone_counter = 0
            self.phone_alert= False
            self.log_lock_phone = False
        #  priortiy status banner
        if self.phone_alert:
            self._play_alarm(self.warning_sound)
            return "Warning: Phone Distraction" , (0 , 0 , 255) 
        elif self.drowsy_alert:
            self._play_alarm(self.alarm_sound)
            return "Warning: Drowsiness Detected" ,(0 , 0 , 255)
        elif self.yawn_alert:
            self._stop_alarm()
            return "Warning: Yawning Detected" ,(0 , 165 , 255)
        else:
            self._stop_alarm()
            return "Driver Active", (0, 255, 0)

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