import streamlit as st
import cv2
import torch
import mediapipe as mp
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

from model.gesture_model import GestureModel

st.title("Gesture Interpreter")

# Load model
model = GestureModel()
model.load_state_dict(torch.load("saved_models/gesture_model.pth"))
model.eval() # Set to evaluation mode (no training)

labels = ["A", "B", "C", "D", "E", "F"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


class GestureProcessor(VideoProcessorBase): # This class processes each video frame
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7
        )

    def recv(self, frame): # Runs for every frame from webcam
        img = frame.to_ndarray(format="bgr24")

        # Mirror for natural feel
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        landmark_list = []

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:

                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

            if len(landmark_list) == 42:
                data = torch.tensor(landmark_list, dtype=torch.float32).unsqueeze(0)
                output = model(data)
                # Convert list → tensor
                # Add batch dimension
                # Pass into model

                predicted = torch.argmax(output).item()
                gesture = labels[predicted]
                # argmax = find index of largest value
                # [0.1, 0.7, 0.05, 0.1, ...] Largest value = 0.7, Index = 1
                # .item() just converts it from tensor → normal number

                cv2.putText(
                    img,
                    f"Gesture: {gesture}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return av.VideoFrame.from_ndarray(img, format="bgr24") # Convert back to video frame for streaming

#ICE/STUN configuration.
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]}
    ]
}

#Opens webcam in browser, Runs your GestureProcessor on each frame
webrtc_streamer(
    key="gesture",
    video_processor_factory=GestureProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15},
        },
        "audio": False,
    },
    async_processing=True,
)
# Webcam → Frame → MediaPipe → Landmarks → Model → Prediction → Display
# streamlit run app.py to start the application.