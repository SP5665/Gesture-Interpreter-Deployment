import cv2
import mediapipe as mp
import csv
import os

# -------------------------------
# Step 1: Setup MediaPipe Hands
# -------------------------------

# Load the MediaPipe Hands solution
mp_hands = mp.solutions.hands

# Utility used to draw landmarks on the screen
mp_draw = mp.solutions.drawing_utils

# Create hand detection object
hands = mp_hands.Hands(
    static_image_mode=False,      # Use continuous detection (video mode)
    max_num_hands=1,              # Detect only one hand
    min_detection_confidence=0.7  # Minimum detection confidence
)

# -------------------------------
# Step 2: Setup CSV file location
# -------------------------------

# Get absolute path of project root
base_dir = os.path.dirname(os.path.dirname(__file__))

# Create path to data/gestures.csv
csv_path = os.path.join(base_dir, "data", "gestures.csv")

# Open CSV file in append mode (adds new rows)
file = open(csv_path, "a", newline="")
writer = csv.writer(file)

print("Saving data to:", csv_path)

# -------------------------------
# Step 3: Ask user for gesture label
# -------------------------------

label = input("Enter gesture label (A/B/C/D/E/F etc): ")

# -------------------------------
# Step 4: Start webcam
# -------------------------------

cap = cv2.VideoCapture(0)

print("Press 's' to save gesture data")
print("Press 'q' to quit program")

# -------------------------------
# Step 5: Main loop
# -------------------------------

while True:

    # Capture frame from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)

    # Convert BGR image to RGB (MediaPipe requires RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame and detect hands
    result = hands.process(rgb)

    # List to store hand landmark coordinates
    landmark_list = []

    # If a hand is detected
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Draw landmarks and hand connections
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract x and y coordinates of each landmark
            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x)
                landmark_list.append(lm.y)

    # Show webcam window
    cv2.imshow("Hand Detection", frame)

    # Read keyboard input
    key = cv2.waitKey(1) & 0xFF

    # -------------------------------
    # Step 6: Save data when 's' is pressed
    # -------------------------------
    if key == ord('s') and landmark_list:
        landmark_list.append(label)   # Add label at the end
        writer.writerow(landmark_list)
        print("Gesture Saved!")

    # -------------------------------
    # Step 7: Exit when 'q' is pressed
    # -------------------------------
    if key == ord('q'):
        print("Program Closed")
        break

# -------------------------------
# Step 8: Cleanup
# -------------------------------

cap.release()           # Release camera
cv2.destroyAllWindows() # Close OpenCV windows
file.close()            # Close CSV file