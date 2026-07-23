import cv2
import mediapipe as mp

from utils.calculations import calculate_mar, calculate_ear

class FaceMeshDetector:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        # specify mediapipe indices
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        self.MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178]

    def process_frame(self, frame):
        h, w, _ = frame.shape

        # mediapipe requires rgb images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # if no face visible then return None values
        if not results.multi_face_landmarks:
            return None, None, None

        # take the first landmarks of the first face detected
        face_landmarks = results.multi_face_landmarks[0]

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

        # return metrics and landmarks locations for draw
        return avg_ear, mar, (left_eye_pts, right_eye_pts, mouth_pts)