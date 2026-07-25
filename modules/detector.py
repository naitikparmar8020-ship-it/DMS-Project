import cv2
import mediapipe as mp
import numpy as np
from utils.calculations import calculate_mar, calculate_ear , get_head_pose , calculate_distance

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # new hand setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands = 2 ,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        # specify mediapipe indices
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178]
        self.HAND_TIP_IDX= 8
        self.HAND_WRTIST_IDX =0


    def process_frame(self, frame):
        h, w, _ = frame.shape

        # mediapipe requires rgb images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)


        # if no face visible then return None values
        if not results.multi_face_landmarks:
            return None, None, None , None , None

        # take the first landmarks of the first face detected
        face_landmarks = results.multi_face_landmarks[0]

        pitch , yaw , roll = self._calculate_head_angles(face_landmarks, w , h)

        # helper function to convert normalized coords (0 to 1) to pixel coords
        def get_pixel_coords(indices):
            points = []
            for idx in indices:
                lm = face_landmarks.landmark[idx]
                points.append((int(lm.x * w), int(lm.y * h)))
            return points

        left_eye_pts = get_pixel_coords(self.LEFT_EYE_IDX)
        right_eye_pts = get_pixel_coords(self.RIGHT_EYE_IDX)
        mouth_pts = get_pixel_coords(self.MOUTH_IDX)

        # calculate ear for both eyes and avg them
        left_ear = calculate_ear(left_eye_pts)
        right_ear = calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # calculate MAR
        mar = calculate_mar(mouth_pts)
        
        hands_results = self.hands.process(rgb_frame)
        hand_pts=[]
        # --- THE FIX: Define a starting distance before the loop ---
        min_distance = float('inf')
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
    

        # return metrics and landmarks locations for draw
        return avg_ear, mar, (pitch , yaw , roll) ,(left_eye_pts, right_eye_pts, mouth_pts) , min_distance
    def _calculate_head_angles(self, face_landmarks, frame_width, frame_height):
        # 1. Grab the 2D points from the webcam
        face_2d = []
        for idx in [1, 152, 33, 263, 61, 291]:
            lm = face_landmarks.landmark[idx]
            face_2d.append([float(lm.x * frame_width), float(lm.y * frame_height)])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        
        # 2. Call your calculations.py file to get the 3D matrix
        face_3d_model, camera_matrix = get_head_pose(frame_width, frame_height)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        # 3. Solve the angles using OpenCV
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d_model, face_2d, camera_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        pitch = angles[0] * 360
        yaw = angles[1] * 360
        roll = angles[2] * 360

        return pitch, yaw, roll