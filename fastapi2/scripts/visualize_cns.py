"""
Real-time CNS Biomarker Visualization

Displays live tracking of:
- Ankle positions (heel strike detection)
- Postural sway (center of mass)
- Tremor analysis (wrist motion)
- Gait state (walking vs stationary)
"""
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Data buffers (store last 300 frames = 10 seconds @ 30fps)
BUFFER_SIZE = 300
ankle_left_y = deque(maxlen=BUFFER_SIZE)
ankle_right_y = deque(maxlen=BUFFER_SIZE)
com_y = deque(maxlen=BUFFER_SIZE)
com_x = deque(maxlen=BUFFER_SIZE)
wrist_left_mag = deque(maxlen=BUFFER_SIZE)
wrist_right_mag = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Heel strike detection
heel_strikes = []
last_strike_time = 0

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

start_time = time.time()
frame_count = 0

print("=== CNS Biomarker Visualizer ===")
print("Press 'q' to quit")
print("\nInstructions:")
print("- Stand still to see postural sway")
print("- Walk in place to see gait analysis")
print("- Hold hands steady to see tremor baseline\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = time.time() - start_time
    timestamps.append(current_time)
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    h, w = frame.shape[:2]
    
    # Create visualization panels
    viz_width = 600
    viz_height = h
    viz_panel = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Draw skeleton on main frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract key landmarks
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        
        # Store ankle Y positions (for gait)
        ankle_left_y.append(left_ankle.y)
        ankle_right_y.append(right_ankle.y)
        
        # Store center of mass (for posture)
        com_y_val = (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y) / 4
        com_x_val = (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x) / 4
        com_y.append(com_y_val)
        com_x.append(com_x_val)
        
        # Store wrist magnitude (for tremor)
        wrist_left_mag.append(np.sqrt(left_wrist.x**2 + left_wrist.y**2))
        wrist_right_mag.append(np.sqrt(right_wrist.x**2 + right_wrist.y**2))
        
        # Detect gait state (walking vs stationary)
        if len(com_x) > 30:
            hip_velocities = np.diff(list(com_x)[-30:])
            mean_vel = np.mean(np.abs(hip_velocities))
            is_walking = mean_vel > 0.005
        else:
            is_walking = False
        
        # Detect heel strikes (simple peak detection)
        if len(ankle_left_y) > 30 and is_walking:
            recent_left = list(ankle_left_y)[-30:]
            if len(recent_left) > 5:
                # Local minimum = heel strike
                if recent_left[-3] < recent_left[-5] and recent_left[-3] < recent_left[-1]:
                    if current_time - last_strike_time > 0.5:  # Min 0.5s between strikes
                        heel_strikes.append(current_time)
                        last_strike_time = current_time
                        # Draw strike indicator on ankles
                        cv2.circle(frame, (int(left_ankle.x * w), int(left_ankle.y * h)), 15, (0, 255, 0), 3)
        
        # Keep only recent strikes
        heel_strikes = [t for t in heel_strikes if current_time - t < 10]
        
        # Calculate metrics
        gait_cv = 0.0
        if len(heel_strikes) > 3:
            strike_intervals = np.diff(heel_strikes)
            gait_cv = np.std(strike_intervals) / np.mean(strike_intervals)
        
        posture_sway = np.std(list(com_y)[-60:]) if len(com_y) > 60 else 0.0
        tremor_power = np.std(list(wrist_left_mag)[-60:]) if len(wrist_left_mag) > 60 else 0.0
        
        # === VISUALIZATION PANEL ===
        y_offset = 20
        
        # Title
        cv2.putText(viz_panel, "CNS BIOMARKERS", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 40
        
        # Gait State
        state_color = (0, 255, 0) if is_walking else (100, 100, 100)
        state_text = "WALKING" if is_walking else "STATIONARY"
        cv2.putText(viz_panel, f"State: {state_text}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1)
        y_offset += 30
        
        # Heel Strikes
        cv2.putText(viz_panel, f"Heel Strikes: {len(heel_strikes)}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        # Gait Variability
        gait_status = "Normal" if 0.02 <= gait_cv <= 0.06 else "Abnormal" if gait_cv > 0.06 else "N/A"
        gait_color = (0, 255, 0) if gait_status == "Normal" else (0, 165, 255) if gait_status == "N/A" else (0, 0, 255)
        cv2.putText(viz_panel, f"Gait CV: {gait_cv:.3f} ({gait_status})", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gait_color, 1)
        y_offset += 35
        
        # Postural Sway
        sway_status = "Normal" if posture_sway < 0.05 else "High"
        sway_color = (0, 255, 0) if sway_status == "Normal" else (0, 165, 255)
        cv2.putText(viz_panel, f"Posture Sway: {posture_sway:.4f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sway_color, 1)
        y_offset += 25
        cv2.putText(viz_panel, f"Status: {sway_status}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sway_color, 1)
        y_offset += 35
        
        # Tremor
        tremor_status = "Normal" if tremor_power < 0.05 else "Elevated"
        tremor_color = (0, 255, 0) if tremor_status == "Normal" else (0, 165, 255)
        cv2.putText(viz_panel, f"Tremor Power: {tremor_power:.4f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tremor_color, 1)
        y_offset += 25
        cv2.putText(viz_panel, f"Status: {tremor_status}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, tremor_color, 1)
        y_offset += 40
        
        # === PLOT ANKLE TRAJECTORY ===
        if len(ankle_left_y) > 10:
            plot_h = 150
            plot_y_start = y_offset
            
            cv2.putText(viz_panel, "Ankle Height (Gait)", (10, plot_y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Normalize and plot
            ankle_data = list(ankle_left_y)[-100:]
            if len(ankle_data) > 1:
                min_val, max_val = min(ankle_data), max(ankle_data)
                if max_val - min_val > 0.01:
                    for i in range(1, len(ankle_data)):
                        x1 = int((i-1) * (viz_width-20) / len(ankle_data)) + 10
                        x2 = int(i * (viz_width-20) / len(ankle_data)) + 10
                        y1 = int(plot_y_start + 20 + (1 - (ankle_data[i-1] - min_val) / (max_val - min_val)) * plot_h)
                        y2 = int(plot_y_start + 20 + (1 - (ankle_data[i] - min_val) / (max_val - min_val)) * plot_h)
                        cv2.line(viz_panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            y_offset += plot_h + 40
        
        # === PLOT POSTURAL SWAY ===
        if len(com_y) > 10:
            plot_h = 150
            plot_y_start = y_offset
            
            cv2.putText(viz_panel, "Postural Sway (COM)", (10, plot_y_start), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            com_data = list(com_y)[-100:]
            if len(com_data) > 1:
                min_val, max_val = min(com_data), max(com_data)
                if max_val - min_val > 0.001:
                    for i in range(1, len(com_data)):
                        x1 = int((i-1) * (viz_width-20) / len(com_data)) + 10
                        x2 = int(i * (viz_width-20) / len(com_data)) + 10
                        y1 = int(plot_y_start + 20 + (1 - (com_data[i-1] - min_val) / (max_val - min_val)) * plot_h)
                        y2 = int(plot_y_start + 20 + (1 - (com_data[i] - min_val) / (max_val - min_val)) * plot_h)
                        cv2.line(viz_panel, (x1, y1), (x2, y2), (255, 0, 255), 2)
    
    # Combine frame and visualization
    # Resize frame to fit screen better
    frame_resized = cv2.resize(frame, (800, 600))
    viz_resized = cv2.resize(viz_panel, (600, 600))
    combined = np.hstack([frame_resized, viz_resized])
    
    # Add FPS
    fps = frame_count / (current_time + 0.001)
    cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('CNS Biomarker Visualization', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()

print("\n=== Session Summary ===")
print(f"Duration: {current_time:.1f}s")
print(f"Frames captured: {frame_count}")
print(f"Heel strikes detected: {len(heel_strikes)}")
if len(heel_strikes) > 3:
    strike_intervals = np.diff(heel_strikes)
    print(f"Gait CV: {np.std(strike_intervals) / np.mean(strike_intervals):.3f}")
