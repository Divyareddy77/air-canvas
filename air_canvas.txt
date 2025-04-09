import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a white canvas
canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255

# Previous coordinates
prev_x, prev_y = 0, 0

# Drawing flag
is_drawing = False

# Stroke history for undo functionality
stroke_history = []
current_stroke = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get hand landmarks
            landmarks = hand_landmarks.landmark
            
            # Check if hand is in drawing position (closed with index finger out)
            if (landmarks[8].y < landmarks[7].y < landmarks[6].y and  # Index finger is extended
                landmarks[12].y > landmarks[11].y and  # Middle finger is bent
                landmarks[16].y > landmarks[15].y and  # Ring finger is bent
                landmarks[20].y > landmarks[19].y):  # Pinky is bent
                
                # Get the coordinates of the index finger tip
                index_finger_tip = landmarks[8]
                x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])
                
                if is_drawing:
                    # Draw line on canvas (in black)
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 2)
                    current_stroke.append((prev_x, prev_y, x, y))
                
                prev_x, prev_y = x, y
                is_drawing = True
            else:
                if is_drawing:
                    # End of stroke
                    stroke_history.append(current_stroke)
                    current_stroke = []
                is_drawing = False
            
            # Draw hand landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Combine frame and canvas
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvas_mask = cv2.threshold(canvas_gray, 254, 255, cv2.THRESH_BINARY)
    
    frame_bg = cv2.bitwise_and(frame, frame, mask=canvas_mask)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(canvas_mask))
    
    combined = cv2.add(frame_bg, canvas_fg)
    
    # Show the result
    cv2.imshow('Air Canvas', combined)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear the canvas
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        stroke_history.clear()
    elif key == ord('z'):
        # Undo last stroke
        if stroke_history:
            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
            stroke_history.pop()
            for stroke in stroke_history:
                for line in stroke:
                    cv2.line(canvas, (line[0], line[1]), (line[2], line[3]), (0, 0, 0), 2)

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()