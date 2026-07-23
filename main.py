import cv2
from modules.detector import FaceMeshDetector


def test_run():
    # 1. Open the webcam
    cap = cv2.VideoCapture(0)
    
    # 2. Initialize your detector
    detector = FaceMeshDetector()
    
    print("Starting test... Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # 3. Pass the frame to your new module
        results = detector.process_frame(frame)
        
        # 4. Unpack the results (handling the case where no face is found)
        if results[0] is not None:
            avg_ear, mar, landmarks = results
            # pitch, yaw, roll = head_pose
            
            # Print the live data to the terminal to verify your math
            print(f"EAR: {avg_ear:.2f} | MAR: {mar:.2f} ")
        else:
            print("No face detected.")

        # Show the raw video feed just so you can see yourself
        cv2.imshow("Test Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_run()