import cv2
import mediapipe as mp

# Load MediaPipe hand module
mp_hands = mp.solutions.hands

# Utility used to draw landmarks on the screen
mp_draw = mp.solutions.drawing_utils

# Create hand detection object
hands = mp_hands.Hands(
    static_image_mode=False,   # video mode (not single images)
    max_num_hands=1,           # detect only one hand
    min_detection_confidence=0.7  # detection confidence threshold
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:

    # Capture frame from webcam
    ret, frame = cap.read()

    # If frame not captured properly stop program
    if not ret:
        break

    # Flip the image (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert image from BGR → RGB (MediaPipe requires RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and detect hand
    result = hands.process(rgb)

    # If hand is detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks and connections on the hand
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Show the webcam window
    cv2.imshow("Hand Detection", frame)

    # Press Q to exit program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()