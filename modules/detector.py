import cv2
import mediapipe as mp
from utils.calculations import calculate_mar, calculate_ear , get_head_pose , calculate_distance

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_path="face_landmarker.task"):
        # Specify mediapipe indices
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178]
        self.HAND_TIP_IDX= 8
        self.HAND_WRTIST_IDX =0


        # Check if running legacy MediaPipe (Python 3.11 / your friend's PC)
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.use_legacy = True
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            # Modern MediaPipe Tasks API (Python 3.14 / your PC)
            self.use_legacy = False
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=max_faces
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

<<<<<<< HEAD
        # -------------------
        # LEGACY API (Python 3.11)
        # -------------------
        if self.use_legacy:
            results = self.face_mesh.process(rgb_frame)
            if not results.multi_face_landmarks:
                return None, None, None
            landmarks_list = results.multi_face_landmarks[0].landmark
=======

        # if no face visible then return None values
        if not results.multi_face_landmarks:
            return None, None, None , None , None
>>>>>>> 342a6c75a84f5eff5acd5759a787660d516e85a8

        # -------------------
        # MODERN TASKS API (Python 3.14)
        # -------------------
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if not results.face_landmarks:
                return None, None, None
            landmarks_list = results.face_landmarks[0]

<<<<<<< HEAD
        # Helper function to convert normalized coords (0 to 1) to pixel coords
=======
        pitch , yaw , roll = self.get_head_pose(face_landmarks, w , h)

        # helper function to convert normalized coords (0 to 1) to pixel coords
>>>>>>> 342a6c75a84f5eff5acd5759a787660d516e85a8
        def get_pixel_coords(indices):
            points = []
            for idx in indices:
                lm = landmarks_list[idx]
                points.append((int(lm.x * w), int(lm.y * h)))
            return points

        left_eye_pts = get_pixel_coords(self.LEFT_EYE_IDX)
        right_eye_pts = get_pixel_coords(self.RIGHT_EYE_IDX)
        mouth_pts = get_pixel_coords(self.MOUTH_IDX)

        # Calculate EAR for both eyes and average them
        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Calculate MAR
        mar = calculate_mar(mouth_pts)
        
        hands_results = self.hands.process(rgb_frame)
        hand_pts=[]
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Grab the index finger tip (MediaPipe point 8)
                index_x = int(hand_landmarks.landmark[8].x * w)
                index_y = int(hand_landmarks.landmark[8].y * h)
                hand_point = (index_x, index_y)

                # Grab the left cheek (MediaPipe point 234) from your face mesh
                cheek_x = int(face_landmarks.landmark[234].x * w)
                cheek_y = int(face_landmarks.landmark[234].y * h)
                cheek_point = (cheek_x, cheek_y)

                dist= calculate_distance(hand_point,cheek_point)

                # Keep the smallest distance (in case both hands are on screen)
                if dist < min_distance:
                    min_distance = dist

<<<<<<< HEAD
        # Return metrics and landmark locations for drawing
        return avg_ear, mar, (left_eye_pts, right_eye_pts, mouth_pts)
=======
        # return metrics and landmarks locations for draw
        return avg_ear, mar, (pitch , yaw , roll) ,(left_eye_pts, right_eye_pts, mouth_pts) , min_distance
    
>>>>>>> 342a6c75a84f5eff5acd5759a787660d516e85a8
