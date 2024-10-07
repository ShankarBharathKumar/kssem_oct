import streamlit as st
import cv2
import dlib
from keras.models import load_model
import numpy as np
import time  # Import time module for timing functionality
import pandas as pd  # Import pandas for DataFrame creation

# Load the pre-trained facial landmark detection model from dlib
landmark_model = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_model)

# Load the pre-trained facial expression recognition model
expression_model = "models/fer2013_mini_XCEPTION.102-0.66.hdf5"
emotion_labels = ["Angry", "Disgust", "Fear", "Smile", "Sad", "Surprise", "Neutral"]
emotion_classifier = load_model(expression_model, compile=False)

# Load the Haar cascade file for face detection
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Streamlit setup
st.title("Facial Expression Recognition App")

# Provide user with options to choose between video or camera
choice = st.sidebar.radio("Choose input source", ('Video', 'Camera'))

if choice == "Video":
    video_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        with open("temp_video.mp4", "wb") as f:
            f.write(video_file.read())
        cap = cv2.VideoCapture("temp_video.mp4")
else:
    # Initialize the camera
    cap = cv2.VideoCapture(0)

# Initialize counters for each feature
total_frames = 0
genuine_smile_frames = 0
fake_smile_frames = 0
angry_frames = 0
disgust_frames = 0
fear_frames = 0
sad_frames = 0
surprise_frames = 0
neutral_frames = 0

# Start processing the video
start_button = st.button("Start ED")

if start_button:
    # Stream video frames
    frame_window = st.image([])

    # Set the time limit for camera processing to 5 seconds
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            # st.write("Failed to capture video")
            break

        total_frames += 1

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            # Predict emotion
            emotion_probabilities = emotion_classifier.predict(np.reshape(cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)), (1, 64, 64, 1)) / 255.0)[0]
            emotion_index = np.argmax(emotion_probabilities)
            emotion = emotion_labels[emotion_index]
            probability = emotion_probabilities[emotion_index]

            # Categorize expressions and count frames
            if emotion == "Smile":
                if probability >= 0.8:
                    genuine_smile_frames += 1
                else:
                    fake_smile_frames += 1
            elif emotion == "Angry":
                angry_frames += 1
            elif emotion == "Disgust":
                disgust_frames += 1
            elif emotion == "Fear":
                fear_frames += 1
            elif emotion == "Sad":
                sad_frames += 1
            elif emotion == "Surprise":
                surprise_frames += 1
            elif emotion == "Neutral":
                neutral_frames += 1

            # Draw a rectangle around the face and display the emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({probability * 100:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Convert the frame from BGR to RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

        # Check if 5 seconds have passed for camera processing
        if choice == "Camera" and (time.time() - start_time) > 5:
            break

    # Release video capture and show results
    cap.release()

    # After processing, calculate and print percentages
    if total_frames > 0:
        st.write("##### RESULTS #####")
        st.write(f"Total frames: {total_frames}")
        st.write(f"Genuine Smile frames: {genuine_smile_frames} ({(genuine_smile_frames / total_frames) * 100:.2f}%)")
        st.write(f"Fake Smile frames: {fake_smile_frames} ({(fake_smile_frames / total_frames) * 100:.2f}%)")
        st.write(f"Angry frames: {angry_frames} ({(angry_frames / total_frames) * 100:.2f}%)")
        st.write(f"Disgust frames: {disgust_frames} ({(disgust_frames / total_frames) * 100:.2f}%)")
        st.write(f"Fear frames: {fear_frames} ({(fear_frames / total_frames) * 100:.2f}%)")
        st.write(f"Sad frames: {sad_frames} ({(sad_frames / total_frames) * 100:.2f}%)")
        st.write(f"Surprise frames: {surprise_frames} ({(surprise_frames / total_frames) * 100:.2f}%)")
        st.write(f"Neutral frames: {neutral_frames} ({(neutral_frames / total_frames) * 100:.2f}%)")

        st.write("##### RESULTS PLOT #####")

        # Prepare data for bar chart
        emotion_counts = {
            "Genuine Smile": genuine_smile_frames,
            "Fake Smile": fake_smile_frames,
            "Angry": angry_frames,
            "Disgust": disgust_frames,
            "Fear": fear_frames,
            "Sad": sad_frames,
            "Surprise": surprise_frames,
            "Neutral": neutral_frames,
        }

        # Create a DataFrame
        df = pd.DataFrame(emotion_counts.items(), columns=["Emotion", "Count"])
        df.set_index("Emotion", inplace=True)

        # Plot bar chart
        st.bar_chart(df)

    else:
        st.write("No frames were processed. Please check the video or camera source.")
