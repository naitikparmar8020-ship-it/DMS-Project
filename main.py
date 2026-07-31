import cv2
from modules.detector import FaceMeshDetector
from modules.tracker import DrowsinessTracker


def test_run():
    # 1. Open the webcam (Using 0 for default camera)
    cap = cv2.VideoCapture(0)
    # this line of code will reduce the resoultion 
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 2. Initialize both the detector (your math) and the tracker (her UI/logic)
    detector = FaceMeshDetector()
    tracker = DrowsinessTracker()
    
    print("Starting test... Press 'q' to quit.")

    while True:

        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # 3. Pass the frame to your math module
        results = detector.process_frame(frame)
        
        # 4. Handle the results (only run if a face is actually found)  
        if results[0] is not None:
            # Unpack the 3 main items you returned
            avg_ear, mar, head_pose , landmarks, hand_distance , hand_pts , phone_detected , phone_box= results
            
            # Unpack the 3 lists of coordinates for the drawing function
            pitch , yaw , roll = head_pose
            left_eye_pts, right_eye_pts, mouth_pts = landmarks
            
            # Feed your numbers into her tracker logic
            status_text, status_color= tracker.update(avg_ear, mar , pitch ,yaw , hand_distance , phone_detected)
            
            # Hand everything to her drawing function to paint the UI onto the frame
            frame = tracker.draw_ui(
                frame, avg_ear, mar, status_text, status_color,
                left_eye_pts, right_eye_pts, mouth_pts , hand_pts , phone_box
            )
            # This data show on your terminal
            # Print the live data to the terminal to verify your math is running
            print(f"EAR: {avg_ear:.2f} | MAR: {mar:.2f} | YAW: {yaw:.2f} | Hand_Dist:{hand_distance:.2f} |Status: {status_text}")
        else:
            print("No face detected.")

        # Show the final painted video feed 
        cv2.imshow("Test Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_run()