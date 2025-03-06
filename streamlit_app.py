import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
from PIL import Image

st.title("🎈 Shoulder Angle Analysis")
st.write(
    "This app analyzes your shoulder angles. Take snapshots to see analysis."
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

def draw_angle_arc(image, hip_point, shoulder_point, elbow_point, angle, color=(0, 200, 0), radius=30, text_offset=(0, 0)):
    """
    Draw an arc to visualize the angle between three points
    Always fills the angle between the trunk and arm
    
    Args:
        image: Image to draw on
        hip_point: Hip point
        shoulder_point: Shoulder point (vertex)
        elbow_point: Elbow point
        angle: The angle in degrees
        color: Color of the arc (BGR)
        radius: Radius of the arc
        text_offset: Offset for the angle text label (x, y)
    """
    # Convert points to numpy arrays
    hip = np.array(hip_point)
    shoulder = np.array(shoulder_point)
    elbow = np.array(elbow_point)
    
    # Calculate vectors from shoulder to hip and shoulder to elbow
    shoulder_to_hip = hip - shoulder
    shoulder_to_elbow = elbow - shoulder
    
    # Calculate unit vectors
    if np.linalg.norm(shoulder_to_hip) == 0 or np.linalg.norm(shoulder_to_elbow) == 0:
        return image  # Cannot draw arc if vectors have zero length
    
    shoulder_to_hip_unit = shoulder_to_hip / np.linalg.norm(shoulder_to_hip)
    shoulder_to_elbow_unit = shoulder_to_elbow / np.linalg.norm(shoulder_to_elbow)
    
    # Create a list of points to make the arc
    arc_points = []
    
    # Number of steps to create a smooth arc
    steps = 40
    
    # Generate a series of points along the arc
    for i in range(steps + 1):
        # Linear interpolation factor
        t = i / steps
        
        # Spherical linear interpolation (SLERP) between the unit vectors
        # This ensures the arc follows the shortest path between the two vectors
        # Formula: slerp(v1, v2, t) = sin((1-t)*angle) / sin(angle) * v1 + sin(t*angle) / sin(angle) * v2
        
        # Dot product between the unit vectors (clamped to avoid numerical issues)
        dot = np.clip(np.dot(shoulder_to_hip_unit, shoulder_to_elbow_unit), -1.0, 1.0)
        
        # Angle between the unit vectors
        omega = math.acos(dot)
        
        # If vectors are too close, use linear interpolation
        if abs(omega) < 1e-6:
            interp_vector = (1 - t) * shoulder_to_hip_unit + t * shoulder_to_elbow_unit
            interp_vector = interp_vector / np.linalg.norm(interp_vector)
        else:
            # SLERP formula
            interp_vector = (math.sin((1-t)*omega) / math.sin(omega)) * shoulder_to_hip_unit + \
                           (math.sin(t*omega) / math.sin(omega)) * shoulder_to_elbow_unit
        
        # Scale by radius and add to shoulder point
        arc_point = shoulder + radius * interp_vector
        arc_points.append((int(arc_point[0]), int(arc_point[1])))
    
    # Create a polygon for shading (including the shoulder point)
    shading_polygon = np.array([shoulder_point] + arc_points, dtype=np.int32)
    
    # Create a copy of the image for the overlay
    overlay = image.copy()
    
    # Draw the filled polygon
    cv2.fillPoly(overlay, [shading_polygon], color)
    
    # Apply the overlay with transparency
    alpha = 0.4  # Transparency factor
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw outline of the arc
    for i in range(len(arc_points) - 1):
        cv2.line(image, arc_points[i], arc_points[i+1], color, 2)
    
    # Draw lines from shoulder to arc endpoints
    cv2.line(image, shoulder_point, arc_points[0], color, 2)
    cv2.line(image, shoulder_point, arc_points[-1], color, 2)
    
    # Draw angle text with a background box for better visibility
    text = f"{angle:.1f} deg"
    
    # Determine a good position for the text - midway around the arc
    midpoint_idx = len(arc_points) // 2
    text_direction = np.array(arc_points[midpoint_idx]) - np.array(shoulder_point)
    
    # Normalize and scale this vector to place text properly
    if np.linalg.norm(text_direction) > 0:
        text_direction = text_direction / np.linalg.norm(text_direction) * (radius * 1.3)
        text_position = np.array(shoulder_point) + text_direction
    else:
        # Fallback if text_direction is zero
        text_position = np.array(shoulder_point) + np.array([radius * 1.3, 0])
    
    # Apply additional offset
    text_x = int(text_position[0]) + text_offset[0]
    text_y = int(text_position[1]) + text_offset[1]
    
    # Get text size for background box
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # Draw background box
    text_bg_x1 = text_x - 5
    text_bg_y1 = text_y - text_size[1] - 5
    text_bg_x2 = text_x + text_size[0] + 5
    text_bg_y2 = text_y + 5
    
    cv2.rectangle(image, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), (255, 255, 255), -1)
    
    # Draw the text
    cv2.putText(image, text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    
    return image

# Initialize tracking variables in session state
if 'left_shoulder_min' not in st.session_state:
    st.session_state.left_shoulder_min = float('inf')
    st.session_state.left_shoulder_max = float('-inf')
    st.session_state.right_shoulder_min = float('inf')
    st.session_state.right_shoulder_max = float('-inf')
    st.session_state.snapshot_count = 0

# Cache the MediaPipe initialization to save memory
@st.cache_resource
def load_pose_model():
    """Cache the MediaPipe pose model to avoid reloading it"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    return mp_pose, mp_drawing, pose

# Get cached pose model
mp_pose, mp_drawing, pose = load_pose_model()

# Function to process a single image frame
@st.cache_data(ttl=300, max_entries=10)  # Cache for 5 minutes, keep up to 10 entries
def cached_process_image(img_bytes):
    """Process image with caching to improve performance and reduce memory usage"""
    # Convert bytes to numpy array
    img = np.array(Image.open(img_bytes))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Process the image using the non-cached function
    return process_image(img)

def process_image(img):
    """
    Process an image to detect pose and calculate shoulder angles
    This function does the actual work and is called by the cached function
    """
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
        
        # Draw trunk and arm lines for visualization
        # Left side
        cv2.line(img_bgr, left_hip_px, left_shoulder_px, (255, 0, 0), 2)  # Trunk line - blue
        cv2.line(img_bgr, left_shoulder_px, left_elbow_px, (0, 255, 0), 2)  # Arm line - green
        
        # Right side
        cv2.line(img_bgr, right_hip_px, right_shoulder_px, (255, 0, 0), 2)  # Trunk line - blue
        cv2.line(img_bgr, right_shoulder_px, right_elbow_px, (0, 255, 0), 2)  # Arm line - green
        
        # Draw angle arcs with shading
        img_bgr = draw_angle_arc(
            img_bgr, 
            left_hip_px, 
            left_shoulder_px, 
            left_elbow_px, 
            left_shoulder_angle, 
            color=(0, 200, 0),  # Green shade
            radius=50,
            text_offset=(0, -20)
        )
        
        img_bgr = draw_angle_arc(
            img_bgr, 
            right_hip_px, 
            right_shoulder_px, 
            right_elbow_px, 
            right_shoulder_angle, 
            color=(0, 200, 0),  # Green shade
            radius=50,
            text_offset=(0, -20)
        )
        
        # Highlight shoulder joints
        cv2.circle(img_bgr, left_shoulder_px, 8, (0, 0, 255), -1)  # Red circle
        cv2.circle(img_bgr, right_shoulder_px, 8, (0, 0, 255), -1)  # Red circle
        
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
        cv2.putText(img_bgr, f"Left: {left_shoulder_angle:.1f} deg ({left_position})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(img_bgr, f"Right: {right_shoulder_angle:.1f} deg ({right_position})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return img_bgr, left_shoulder_angle, right_shoulder_angle

# Function to get position name
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

# Create layout for the app
col1, col2 = st.columns([2, 1])

with col2:
    # Reset button to clear the min/max values
    if st.button("Reset Range Measurements"):
        st.session_state.left_shoulder_min = float('inf')
        st.session_state.left_shoulder_max = float('-inf')
        st.session_state.right_shoulder_min = float('inf')
        st.session_state.right_shoulder_max = float('-inf')
        st.session_state.snapshot_count = 0
        
        # Clear the cache when resetting to free memory
        cached_process_image.clear()
        
        st.rerun()
    
    # Display range data
    st.subheader("Range of Motion")
    st.write(f"Left Shoulder: {st.session_state.left_shoulder_min:.1f} - {st.session_state.left_shoulder_max:.1f} deg")
    st.write(f"Right Shoulder: {st.session_state.right_shoulder_min:.1f} - {st.session_state.right_shoulder_max:.1f} deg")
    
    # Display snapshot count
    st.write(f"Snapshots taken: {st.session_state.snapshot_count}")
    
    # Add button to clear cache
    if st.button("Clear Cache"):
        cached_process_image.clear()
        st.success("Cache cleared successfully!")
    
    st.subheader("Instructions")
    st.write("""
    1. Allow camera access when prompted
    2. Stand back so your upper body is visible
    3. Take snapshots by clicking the camera button
    4. Each snapshot will be analyzed to show shoulder angles
    5. Click "Reset" to start a new session
    6. If app becomes slow, try clicking "Clear Cache"
    """)
    
    # Color legend
    st.subheader("Color Legend")
    st.markdown("""
    - <span style='color:blue'>Blue lines</span>: Trunk (hip to shoulder)
    - <span style='color:green'>Green lines</span>: Arms (shoulder to elbow)
    - <span style='color:red'>Red circles</span>: Shoulder joints
    - <span style='color:green'>Green shading</span>: Angle measurement
    """, unsafe_allow_html=True)
    
    # Angle reference
    st.subheader("Angle Reference")
    st.markdown("""
    - **0-30 deg**: Arms Up
    - **30-60 deg**: High Position
    - **60-120 deg**: Horizontal
    - **120-150 deg**: Low Position
    - **150-180 deg**: Arms Down
    """)

with col1:
    st.subheader("Shoulder Angle Analysis")
    
    # Camera input
    camera_image = st.camera_input("Take a snapshot")
    
    # Process the image if available
    if camera_image is not None:
        # Increment snapshot count
        st.session_state.snapshot_count += 1
        
        # Process the image using the cached function
        processed_img, left_angle, right_angle = cached_process_image(camera_image)
        
        # Display the processed image
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                 caption=f"Pose Analysis (Snapshot #{st.session_state.snapshot_count})")
        
        # Display current angles
        left_position = get_position(left_angle)
        right_position = get_position(right_angle)
        
        st.markdown(f"""
        ### Current Shoulder Angles:
        - Left Shoulder: **{left_angle:.1f} deg** ({left_position})
        - Right Shoulder: **{right_angle:.1f} deg** ({right_position})
        """)