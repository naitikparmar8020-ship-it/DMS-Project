import math 
import numpy as np
def calculate_ear(eye_points):

    # vertical distance
    v1=math.dist(eye_points[1],eye_points[5])
    v2=math.dist(eye_points[2],eye_points[4])

    #horizontal distance

    h=math.dist(eye_points[0],eye_points[3])


    #EAR formula
    ear=(v1+v2)/(2.0*h)
    return ear

def calculate_mar(mouth_points):

    # vertical distance
    v=math.dist(mouth_points[1],mouth_points[7])

    # horizontal distance
    h=math.dist(mouth_points[0],mouth_points[4])

    # mar formula 
    mar= v/h
    return mar

def get_head_pose(frame_width,frame_height):
    face_3d_model = np.array([
        (0.0 , 0.0 , 0.0 ) , #for nose tip
        (0.0 , -330.0 , -65.0 ), #chin
        (-225.0 , 170.0 , 135.0) , #ledt eye corner  
        (225.0 , 170.0 , -135.0), #right eye
        (-150.0 , -150.0 , -150.0), #left mouth 
        (150.0 , -125.0 , -150.0)  #right mouth
    ],dtype=np.float64)

    # Calculate camera matrix using the width and height passed into the function
    focal_length = frame_width
    center = (frame_width /2 , frame_height /2)
    camera_matrix = np.array([
        [focal_length, 0 , center[0]],
        [0 , focal_length , center[1]],
        [0, 0 , 1]
    ],dtype=np.float64)
    return face_3d_model , camera_matrix
