import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

st.title("🎈 Shoulder Angle Analysis")
st.write(
    "This app tracks your shoulder angles in real-time. Allow camera access when prompted."
)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees)
    Points a, b, c where b is the middle point (vertex)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # Convert to degrees
    angle_degrees = np.degrees(angle)

    return angle_degrees

# Initialize tracking variables in session state
if 'left_shoulder_min' not in st.session_state:
    st.session_state.left_shoulder_min = float('inf')
    st.session_state.left_shoulder_max = float('-inf')
    st.session_state.right_shoulder_min = float('inf')
    st.session_state.right_shoulder_max = float('-inf')

# Create placeholders for displaying data
angle_display = st.empty()
range_display = st.empty()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,  # Set to True for images
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Function to process a single image frame
def process_image(img):
    # Convert the image to RGB (MediaPipe requires RGB input)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe Pose
    results = pose.process(img_rgb)
    
    # Convert back to BGR for OpenCV operations
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    h, w, c = img_bgr.shape
    
    left_shoulder_angle = 0
    right_shoulder_angle = 0
    
    # Draw pose landmarks on the image if detected
    if results.pose_landmarks:
        # Draw the pose landmarks
        mp_drawing.draw_landmarks(
            img_bgr,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        # Get landmarks for shoulder angle calculation
        landmarks = results.pose_landmarks.landmark
        
        # Left side landmarks
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        
        # Right side landmarks
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        # Convert to pixel coordinates for visualization
        left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
        left_shoulder_px = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        left_elbow_px = (int(left_elbow.x * w), int(left_elbow.y * h))
        
        right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))
        right_shoulder_px = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        right_elbow_px = (int(right_elbow.x * w), int(right_elbow.y * h))
        
        # Calculate shoulder angles
        # For left shoulder - angle between hip, shoulder, and elbow
        left_shoulder_angle = calculate_angle(
            [left_hip.x, left_hip.y],
            [left_shoulder.x, left_shoulder.y],
            [left_elbow.x, left_elbow.y]
        )
        
        # For right shoulder - angle between hip, shoulder, and elbow
        right_shoulder_angle = calculate_angle(
            [right_hip.x, right_hip.y],
            [right_shoulder.x, right_shoulder.y],
            [right_elbow.x, right_elbow.y]
        )
        
        # Update min/max values
        st.session_state.left_shoulder_min = min(st.session_state.left_shoulder_min, left_shoulder_angle)
        st.session_state.left_shoulder_max = max(st.session_state.left_shoulder_max, left_shoulder_angle)
        st.session_state.right_shoulder_min = min(st.session_state.right_shoulder_min, right_shoulder_angle)
        st.session_state.right_shoulder_max = max(st.session_state.right_shoulder_max, right_shoulder_angle)
        
        # Draw trunk and arm lines for visualization
        # Left side
        cv2.line(img_bgr, left_hip_px, left_shoulder_px, (255, 0, 0), 2)  # Trunk line - blue
        cv2.line(img_bgr, left_shoulder_px, left_elbow_px, (0, 255, 0), 2)  # Arm line - green
        
        # Right side
        cv2.line(img_bgr, right_hip_px, right_shoulder_px, (255, 0, 0), 2)  # Trunk line - blue
        cv2.line(img_bgr, right_shoulder_px, right_elbow_px, (0, 255, 0), 2)  # Arm line - green
        
        # Highlight shoulder joints
        cv2.circle(img_bgr, left_shoulder_px, 8, (0, 0, 255), -1)  # Red circle
        cv2.circle(img_bgr, right_shoulder_px, 8, (0, 0, 255), -1)  # Red circle
        
        # Display angles directly at the shoulder joints
        cv2.putText(img_bgr, f"{left_shoulder_angle:.1f}°",
                    (left_shoulder_px[0] - 45, left_shoulder_px[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img_bgr, f"{right_shoulder_angle:.1f}°",
                    (right_shoulder_px[0] + 15, right_shoulder_px[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Determine position description
        def get_position(angle):
            if angle < 30:
                return "Arms Up"
            elif angle < 60:
                return "High Position"
            elif angle < 120:
                return "Horizontal"
            elif angle < 150:
                return "Low Position"
            else:
                return "Arms Down"
        
        left_position = get_position(left_shoulder_angle)
        right_position = get_position(right_shoulder_angle)
        
        # Display angle info at the top of the frame
        cv2.putText(img_bgr, f"Left: {left_shoulder_angle:.1f}° ({left_position})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img_bgr, f"Right: {right_shoulder_angle:.1f}° ({right_position})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img_bgr, left_shoulder_angle, right_shoulder_angle

# Create camera input
camera_image = st.camera_input("Enable webcam")

if camera_image is not None:
    # Convert the image to a numpy array for processing
    img = np.array(Image.open(camera_image))
    
    # OpenCV expects BGR format, but Image.open gives RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Process the image
    processed_img, left_angle, right_angle = process_image(img)
    
    # Display the processed image (convert back to RGB for st.image)
    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    
    # Determine position descriptions
    def get_position(angle):
        if angle < 30:
            return "Arms Up"
        elif angle < 60:
            return "High Position"
        elif angle < 120:
            return "Horizontal"
        elif angle < 150:
            return "Low Position"
        else:
            return "Arms Down"
    
    left_position = get_position(left_angle)
    right_position = get_position(right_angle)
    
    # Display current angles
    st.markdown(f"""
    ### Current Angles:
    - Left Shoulder: **{left_angle:.1f}°** ({left_position})
    - Right Shoulder: **{right_angle:.1f}°** ({right_position})
    """)
    
    # Display min/max ranges
    st.markdown(f"""
    ### Range of Motion:
    - Left Shoulder Range: **{st.session_state.left_shoulder_min:.1f}° - {st.session_state.left_shoulder_max:.1f}°**
    - Right Shoulder Range: **{st.session_state.right_shoulder_min:.1f}° - {st.session_state.right_shoulder_max:.1f}°**
    """)

# Reset button to clear the min/max values
if st.button("Reset Range Measurements"):
    st.session_state.left_shoulder_min = float('inf')
    st.session_state.left_shoulder_max = float('-inf')
    st.session_state.right_shoulder_min = float('inf')
    st.session_state.right_shoulder_max = float('-inf')
    st.rerun()  # Updated from experimental_rerun()

st.markdown("""
### Instructions
1. Allow camera access when prompted
2. Click the "Enable webcam" button to take a snapshot
3. Stand back so your upper body is visible
4. Move your arms to different positions and take multiple snapshots to test range of motion
5. Click "Reset Range Measurements" to start a new session
""")