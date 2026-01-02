import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- NUCLEAR FIX: DIRECT IMPORTS ---
# We look for the tools directly where they live, bypassing the broken shortcuts.
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose

# --- HELPER FUNCTION ---
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

# --- VIDEO PROCESSOR CLASS ---
class RehabProcessor(VideoTransformerBase):
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None
        self.feedback = "Fix Form"

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # 1. Process Image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        
        # 2. Draw Landmarks & Logic
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Left Arm Coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Curl Logic
            if angle > 160:
                self.stage = "down"
                self.feedback = "Good Extension"
            if angle < 30 and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
                self.feedback = "Good Curl!"

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Draw Data
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, self.feedback, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- APP LAYOUT ---
st.title("Rehab Tracker Mobile")
st.write("Ensure you allow camera access when prompted.")
webrtc_streamer(key="rehab", video_processor_factory=RehabProcessor)
