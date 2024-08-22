import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw_color = (255, 255, 255)  # Color for drawing
erase_color = (0, 0, 0)  # Color for erasing

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a blank canvas to draw
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize previous position variables
prev_x, prev_y = 0, 0


# Function to draw lines on canvas
def draw_line(canvas, start, end, color, thickness=2):
    cv2.line(canvas, start, end, color, thickness)


# Function to erase drawn areas on canvas
def erase_area(canvas, center, radius, color):
    cv2.circle(canvas, center, radius, color, -1)


# Main loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    results = hands.process(frame_rgb)

    # Draw landmarks and get hand positions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks of the hand
            landmarks = hand_landmarks.landmark

            # Get the coordinates of the index finger tip
            index_tip_x, index_tip_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]), int(
                landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Get the coordinates of the palm (center of the hand)
            palm_x, palm_y = int(landmarks[mp_hands.HandLandmark.WRIST].x * frame.shape[1]), int(
                landmarks[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Check if the left hand and palm are detected
            if results.multi_handedness[0].classification[0].label == 'Left' and landmarks[
                mp_hands.HandLandmark.WRIST].x < 0.5:
                # Erase with the palm
                erase_area(canvas, (palm_x, palm_y), 140, erase_color)
            else:
                # Draw with the index finger
                if prev_x != 0 and prev_y != 0:
                    draw_line(canvas, (prev_x, prev_y), (index_tip_x, index_tip_y), draw_color)
                prev_x, prev_y = index_tip_x, index_tip_y

    # Display frame and canvas
    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
