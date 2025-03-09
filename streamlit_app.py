import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from PIL import Image

st.title("🎈 Joint Angle Analysis")
st.write(
    "This app analyzes joint angles from snapshots. Select which joint to analyze."
)

# Add a selection for which joint to analyze
analysis_type = st.radio(
    "Select joint to analyze:",
    ["Shoulder Angles", "Knee Angles", "Ankle Angles"],
    horizontal=True
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

def draw_angle_arc(image, point1, point2, point3, angle, color=(0, 200, 0), radius=30, text_offset=(0, 0)):
    """
    Draw an arc to visualize the angle between three points
    Always fills the angle between the vectors
    
    Args:
        image: Image to draw on
        point1: First point (e.g., hip or knee)
        point2: Vertex point (e.g., shoulder or ankle)
        point3: Third point (e.g., elbow or foot)
        angle: The angle in degrees
        color: Color of the arc (BGR)
        radius: Radius of the arc
        text_offset: Offset for the angle text label (x, y)
    """
    # Convert points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)
    p3 = np.array(point3)
    
    # Calculate vectors from vertex to other points
    v1 = p1 - p2
    v2 = p3 - p2
    
    # Calculate unit vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return image  # Cannot draw arc if vectors have zero length
    
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    
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
        dot = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
        
        # Angle between the unit vectors
        omega = math.acos(dot)
        
        # If vectors are too close, use linear interpolation
        if abs(omega) < 1e-6:
            interp_vector = (1 - t) * v1_unit + t * v2_unit
            interp_vector = interp_vector / np.linalg.norm(interp_vector)
        else:
            # SLERP formula
            interp_vector = (math.sin((1-t)*omega) / math.sin(omega)) * v1_unit + \
                           (math.sin(t*omega) / math.sin(omega)) * v2_unit
        
        # Scale by radius and add to vertex point
        arc_point = p2 + radius * interp_vector
        arc_points.append((int(arc_point[0]), int(arc_point[1])))
    
    # Create a polygon for shading (including the vertex point)
    shading_polygon = np.array([point2] + arc_points, dtype=np.int32)
    
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
    
    # Draw lines from vertex to arc endpoints
    cv2.line(image, point2, arc_points[0], color, 2)
    cv2.line(image, point2, arc_points[-1], color, 2)
    
    # Draw angle text with a background box for better visibility
    text = f"{angle:.1f} deg"
    
    # Determine a good position for the text - midway around the arc
    midpoint_idx = len(arc_points) // 2
    text_direction = np.array(arc_points[midpoint_idx]) - np.array(point2)
    
    # Normalize and scale this vector to place text properly
    if np.linalg.norm(text_direction) > 0:
        text_direction = text_direction / np.linalg.norm(text_direction) * (radius * 1.3)
        text_position = np.array(point2) + text_direction
    else:
        # Fallback if text_direction is zero
        text_position = np.array(point2) + np.array([radius * 1.3, 0])
    
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
    st.session_state.left_knee_min = float('inf')
    st.session_state.left_knee_max = float('-inf')
    st.session_state.right_knee_min = float('inf')
    st.session_state.right_knee_max = float('-inf')
    st.session_state.left_ankle_min = float('inf')
    st.session_state.left_ankle_max = float('-inf')
    st.session_state.right_ankle_min = float('inf')
    st.session_state.right_ankle_max = float('-inf')
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
def cached_process_image(img_bytes, analysis_type):
    """Process image with caching to improve performance and reduce memory usage"""
    # Convert bytes to numpy array
    img = np.array(Image.open(img_bytes))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Process the image using the non-cached function
    return process_image(img, analysis_type)

def process_image(img, analysis_type):
    """
    Process an image to detect pose and calculate joint angles
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
    
    # Initialize angle variables
    left_shoulder_angle = 0
    right_shoulder_angle = 0
    left_knee_angle = 0
    right_knee_angle = 0
    left_ankle_angle = 0
    right_ankle_angle = 0
    
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
        
        # Get landmarks for angle calculation
        landmarks = results.pose_landmarks.landmark
        
        if analysis_type == "Shoulder Angles":
            # Left side shoulder landmarks
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            
            # Right side shoulder landmarks
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
            
            # Update shoulder min/max values
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
            
            # Determine position description for shoulders
            def get_shoulder_position(angle):
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
            
            left_position = get_shoulder_position(left_shoulder_angle)
            right_position = get_shoulder_position(right_shoulder_angle)
            
            # Display angle info at the top of the frame
            cv2.putText(img_bgr, f"Left: {left_shoulder_angle:.1f} deg ({left_position})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img_bgr, f"Right: {right_shoulder_angle:.1f} deg ({right_position})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        elif analysis_type == "Knee Angles":
            # Left side knee landmarks
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Right side knee landmarks
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            # Convert to pixel coordinates for visualization
            left_hip_px = (int(left_hip.x * w), int(left_hip.y * h))
            left_knee_px = (int(left_knee.x * w), int(left_knee.y * h))
            left_ankle_px = (int(left_ankle.x * w), int(left_ankle.y * h))
            
            right_hip_px = (int(right_hip.x * w), int(right_hip.y * h))
            right_knee_px = (int(right_knee.x * w), int(right_knee.y * h))
            right_ankle_px = (int(right_ankle.x * w), int(right_ankle.y * h))
            
            # Calculate knee angles
            # For left knee - angle between hip, knee, and ankle
            left_knee_angle = calculate_angle(
                [left_hip.x, left_hip.y],
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y]
            )
            
            # For right knee - angle between hip, knee, and ankle
            right_knee_angle = calculate_angle(
                [right_hip.x, right_hip.y],
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y]
            )
            
            # Update knee min/max values
            st.session_state.left_knee_min = min(st.session_state.left_knee_min, left_knee_angle)
            st.session_state.left_knee_max = max(st.session_state.left_knee_max, left_knee_angle)
            st.session_state.right_knee_min = min(st.session_state.right_knee_min, right_knee_angle)
            st.session_state.right_knee_max = max(st.session_state.right_knee_max, right_knee_angle)
            
            # Draw thigh and lower leg lines for visualization
            # Left side
            cv2.line(img_bgr, left_hip_px, left_knee_px, (255, 0, 0), 2)  # Thigh line - blue
            cv2.line(img_bgr, left_knee_px, left_ankle_px, (0, 255, 0), 2)  # Lower leg line - green
            
            # Right side
            cv2.line(img_bgr, right_hip_px, right_knee_px, (255, 0, 0), 2)  # Thigh line - blue
            cv2.line(img_bgr, right_knee_px, right_ankle_px, (0, 255, 0), 2)  # Lower leg line - green
            
            # Draw angle arcs with shading
            img_bgr = draw_angle_arc(
                img_bgr, 
                left_hip_px, 
                left_knee_px, 
                left_ankle_px, 
                left_knee_angle, 
                color=(0, 200, 0),  # Green shade
                radius=50,
                text_offset=(0, -20)
            )
            
            img_bgr = draw_angle_arc(
                img_bgr, 
                right_hip_px, 
                right_knee_px, 
                right_ankle_px, 
                right_knee_angle, 
                color=(0, 200, 0),  # Green shade
                radius=50,
                text_offset=(0, -20)
            )
            
            # Highlight knee joints
            cv2.circle(img_bgr, left_knee_px, 8, (0, 0, 255), -1)  # Red circle
            cv2.circle(img_bgr, right_knee_px, 8, (0, 0, 255), -1)  # Red circle
            
            # Determine knee position description
            def get_knee_position(angle):
                if angle < 100:
                    return "Deep Flexion"
                elif angle < 140:
                    return "Moderate Flexion"
                elif angle < 170:
                    return "Slight Flexion"
                else:
                    return "Extended"
            
            left_position = get_knee_position(left_knee_angle)
            right_position = get_knee_position(right_knee_angle)
            
            # Display angle info at the top of the frame
            cv2.putText(img_bgr, f"Left Knee: {left_knee_angle:.1f} deg ({left_position})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img_bgr, f"Right Knee: {right_knee_angle:.1f} deg ({right_position})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
        elif analysis_type == "Ankle Angles":
            # Left side ankle landmarks
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            
            # Right side ankle landmarks
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
            right_foot_index = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            
            # Convert to pixel coordinates for visualization
            left_knee_px = (int(left_knee.x * w), int(left_knee.y * h))
            left_ankle_px = (int(left_ankle.x * w), int(left_ankle.y * h))
            left_foot_index_px = (int(left_foot_index.x * w), int(left_foot_index.y * h))
            
            right_knee_px = (int(right_knee.x * w), int(right_knee.y * h))
            right_ankle_px = (int(right_ankle.x * w), int(right_ankle.y * h))
            right_foot_index_px = (int(right_foot_index.x * w), int(right_foot_index.y * h))
            
            # Calculate ankle angles
            # For left ankle - angle between knee, ankle, and foot index
            left_ankle_angle = calculate_angle(
                [left_knee.x, left_knee.y],
                [left_ankle.x, left_ankle.y],
                [left_foot_index.x, left_foot_index.y]
            )
            
            # For right ankle - angle between knee, ankle, and foot index
            right_ankle_angle = calculate_angle(
                [right_knee.x, right_knee.y],
                [right_ankle.x, right_ankle.y],
                [right_foot_index.x, right_foot_index.y]
            )
            
            # Update ankle min/max values
            st.session_state.left_ankle_min = min(st.session_state.left_ankle_min, left_ankle_angle)
            st.session_state.left_ankle_max = max(st.session_state.left_ankle_max, left_ankle_angle)
            st.session_state.right_ankle_min = min(st.session_state.right_ankle_min, right_ankle_angle)
            st.session_state.right_ankle_max = max(st.session_state.right_ankle_max, right_ankle_angle)
            
            # Draw leg and foot lines for visualization
            # Left side
            cv2.line(img_bgr, left_knee_px, left_ankle_px, (255, 0, 0), 2)  # Leg line - blue
            cv2.line(img_bgr, left_ankle_px, left_foot_index_px, (0, 255, 0), 2)  # Foot line - green
            
            # Right side
            cv2.line(img_bgr, right_knee_px, right_ankle_px, (255, 0, 0), 2)  # Leg line - blue
            cv2.line(img_bgr, right_ankle_px, right_foot_index_px, (0, 255, 0), 2)  # Foot line - green
            
            # Draw angle arcs with shading
            img_bgr = draw_angle_arc(
                img_bgr, 
                left_knee_px, 
                left_ankle_px, 
                left_foot_index_px, 
                left_ankle_angle, 
                color=(0, 200, 0),  # Green shade
                radius=50,
                text_offset=(0, -20)
            )
            
            img_bgr = draw_angle_arc(
                img_bgr, 
                right_knee_px, 
                right_ankle_px, 
                right_foot_index_px, 
                right_ankle_angle, 
                color=(0, 200, 0),  # Green shade
                radius=50,
                text_offset=(0, -20)
            )
            
            # Highlight ankle joints
            cv2.circle(img_bgr, left_ankle_px, 8, (0, 0, 255), -1)  # Red circle
            cv2.circle(img_bgr, right_ankle_px, 8, (0, 0, 255), -1)  # Red circle
            
            # Determine ankle position description
            def get_ankle_position(angle):
                if angle < 80:
                    return "Dorsiflexion"
                elif angle < 100:
                    return "Neutral"
                else:
                    return "Plantarflexion"
            
            left_position = get_ankle_position(left_ankle_angle)
            right_position = get_ankle_position(right_ankle_angle)
            
            # Display angle info at the top of the frame
            cv2.putText(img_bgr, f"Left Ankle: {left_ankle_angle:.1f} deg ({left_position})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.putText(img_bgr, f"Right Ankle: {right_ankle_angle:.1f} deg ({right_position})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    if analysis_type == "Shoulder Angles":
        return img_bgr, left_shoulder_angle, right_shoulder_angle
    elif analysis_type == "Knee Angles":
        return img_bgr, left_knee_angle, right_knee_angle
    else:  # Ankle Angles
        return img_bgr, left_ankle_angle, right_ankle_angle

# Create layout for the app
col1, col2 = st.columns([2, 1])

with col2:
    # Reset button to clear the min/max values
    if st.button("Reset Measurements"):
        if analysis_type == "Shoulder Angles":
            st.session_state.left_shoulder_min = float('inf')
            st.session_state.left_shoulder_max = float('-inf')
            st.session_state.right_shoulder_min = float('inf')
            st.session_state.right_shoulder_max = float('-inf')
        elif analysis_type == "Knee Angles":
            st.session_state.left_knee_min = float('inf')
            st.session_state.left_knee_max = float('-inf')
            st.session_state.right_knee_min = float('inf')
            st.session_state.right_knee_max = float('-inf')
        else:  # Ankle Angles
            st.session_state.left_ankle_min = float('inf')
            st.session_state.left_ankle_max = float('-inf')
            st.session_state.right_ankle_min = float('inf')
            st.session_state.right_ankle_max = float('-inf')
            
        st.session_state.snapshot_count = 0
        
        # Clear the cache when resetting to free memory
        cached_process_image.clear()
        
        st.rerun()
    
    # Display range data
    st.subheader("Range of Motion")
    
    if analysis_type == "Shoulder Angles":
        st.write(f"Left Shoulder: {st.session_state.left_shoulder_min:.1f} - {st.session_state.left_shoulder_max:.1f} deg")
        st.write(f"Right Shoulder: {st.session_state.right_shoulder_min:.1f} - {st.session_state.right_shoulder_max:.1f} deg")
    elif analysis_type == "Knee Angles":
        st.write(f"Left Knee: {st.session_state.left_knee_min:.1f} - {st.session_state.left_knee_max:.1f} deg")
        st.write(f"Right Knee: {st.session_state.right_knee_min:.1f} - {st.session_state.right_knee_max:.1f} deg")
    else:  # Ankle Angles
        st.write(f"Left Ankle: {st.session_state.left_ankle_min:.1f} - {st.session_state.left_ankle_max:.1f} deg")
        st.write(f"Right Ankle: {st.session_state.right_ankle_min:.1f} - {st.session_state.right_ankle_max:.1f} deg")
    
    # Display snapshot count
    st.write(f"Snapshots taken: {st.session_state.snapshot_count}")
    
    # Add button to clear cache
    if st.button("Clear Cache"):
        cached_process_image.clear()
        st.success("Cache cleared successfully!")
    
    st.subheader("Instructions")
    
    if analysis_type == "Shoulder Angles":
        st.write("""
        1. Allow camera access when prompted
        2. Stand facing the camera
        3. Make sure your upper body is visible
        4. Take snapshots by clicking the camera button
        5. Click "Reset" to start a new session
        6. If app becomes slow, try clicking "Clear Cache"
        """)
        
        # Color legend for shoulder
        st.subheader("Color Legend")
        st.markdown("""
        - <span style='color:blue'>Blue lines</span>: Trunk (hip to shoulder)
        - <span style='color:green'>Green lines</span>: Arms (shoulder to elbow)
        - <span style='color:red'>Red circles</span>: Shoulder joints
        - <span style='color:green'>Green shading</span>: Angle measurement
        """, unsafe_allow_html=True)
        
        # Angle reference for shoulder
        st.subheader("Angle Reference")
        st.markdown("""
        - **0-30 deg**: Arms Up
        - **30-60 deg**: High Position
        - **60-120 deg**: Horizontal
        - **120-150 deg**: Low Position
        - **150-180 deg**: Arms Down
        """)
    elif analysis_type == "Knee Angles":
        st.write("""
        1. Allow camera access when prompted
        2. Position yourself so your full legs are visible
        3. For best results, stand in profile (side view)
        4. Take snapshots by clicking the camera button
        5. Click "Reset" to start a new session
        6. If app becomes slow, try clicking "Clear Cache"
        """)
        
        # Color legend for knee
        st.subheader("Color Legend")
        st.markdown("""
        - <span style='color:blue'>Blue lines</span>: Thigh (hip to knee)
        - <span style='color:green'>Green lines</span>: Lower leg (knee to ankle)
        - <span style='color:red'>Red circles</span>: Knee joints
        - <span style='color:green'>Green shading</span>: Angle measurement
        """, unsafe_allow_html=True)
        
        # Angle reference for knee
        st.subheader("Angle Reference")
        st.markdown("""
        - **< 100 deg**: Deep Flexion
        - **100-140 deg**: Moderate Flexion
        - **140-170 deg**: Slight Flexion
        - **> 170 deg**: Extended
        """)
    else:  # Ankle Angles
        st.write("""
        1. Allow camera access when prompted
        2. Position yourself so your legs and feet are visible
        3. For best results, stand in profile (side view)
        4. Take snapshots by clicking the camera button
        5. Click "Reset" to start a new session
        6. If app becomes slow, try clicking "Clear Cache"
        """)
        
        # Color legend for ankle
        st.subheader("Color Legend")
        st.markdown("""
        - <span style='color:blue'>Blue lines</span>: Leg (knee to ankle)
        - <span style='color:green'>Green lines</span>: Foot (ankle to toe)
        - <span style='color:red'>Red circles</span>: Ankle joints
        - <span style='color:green'>Green shading</span>: Angle measurement
        """, unsafe_allow_html=True)
        
        # Angle reference for ankle
        st.subheader("Angle Reference")
        st.markdown("""
        - **< 80 deg**: Dorsiflexion (toes up)
        - **80-100 deg**: Neutral position
        - **> 100 deg**: Plantarflexion (pointing toes)
        """)
with col1:
    if analysis_type == "Shoulder Angles":
        st.subheader("Shoulder Angle Analysis")
    elif analysis_type == "Knee Angles":
        st.subheader("Knee Angle Analysis")
    else:  # Ankle Angles
        st.subheader("Ankle Angle Analysis")
    
    # Camera input
    camera_image = st.camera_input("Take a snapshot")
    
    # Process the image if available
    if camera_image is not None:
        # Increment snapshot count
        st.session_state.snapshot_count += 1
        
        # Process the image using the cached function with the selected analysis type
        processed_img, left_angle, right_angle = cached_process_image(camera_image, analysis_type)
        
        # Display the processed image
        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                 caption=f"Analysis (Snapshot #{st.session_state.snapshot_count})")
        
        # Display current angles
        if analysis_type == "Shoulder Angles":
            # Define shoulder position function here too for access in this scope
            def get_shoulder_position(angle):
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
                
            left_position = get_shoulder_position(left_angle)
            right_position = get_shoulder_position(right_angle)
            
            st.markdown(f"""
            ### Current Shoulder Angles:
            - Left Shoulder: **{left_angle:.1f} deg** ({left_position})
            - Right Shoulder: **{right_angle:.1f} deg** ({right_position})
            """)
        elif analysis_type == "Knee Angles":
            # Define knee position function here too for access in this scope
            def get_knee_position(angle):
                if angle < 100:
                    return "Deep Flexion"
                elif angle < 140:
                    return "Moderate Flexion"
                elif angle < 170:
                    return "Slight Flexion"
                else:
                    return "Extended"
                
            left_position = get_knee_position(left_angle)
            right_position = get_knee_position(right_angle)
            
            st.markdown(f"""
            ### Current Knee Angles:
            - Left Knee: **{left_angle:.1f} deg** ({left_position})
            - Right Knee: **{right_angle:.1f} deg** ({right_position})
            """)
        else:  # Ankle Angles
            # Define ankle position function here too for access in this scope
            def get_ankle_position(angle):
                if angle < 80:
                    return "Dorsiflexion"
                elif angle < 100:
                    return "Neutral"
                else:
                    return "Plantarflexion"
                
            left_position = get_ankle_position(left_angle)
            right_position = get_ankle_position(right_angle)
            
            st.markdown(f"""
            ### Current Ankle Angles:
            - Left Ankle: **{left_angle:.1f} deg** ({left_position})
            - Right Ankle: **{right_angle:.1f} deg** ({right_position})
            """)