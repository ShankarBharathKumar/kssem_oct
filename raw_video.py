import cv2
import dlib
from keras.models import load_model
import numpy as np
import os

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

# Initialize the tracker
tracker = dlib.correlation_tracker()

# Load the video capture from a video file
video_folder = 'video'
video_files = os.listdir(video_folder)

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

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

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Increment total frames counter
        total_frames += 1

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Update the tracker with the current face region
            tracker.start_track(frame, dlib.rectangle(x, y, x+w, y+h))

            # Convert the frame to RGB for facial landmark detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Update the tracker with the new frame
            tracker.update(rgb)

            # Get the updated position of the face
            rect = tracker.get_position()
            x = int(rect.left())
            y = int(rect.top())
            w = int(rect.width())
            h = int(rect.height())

            # Extract the region of interest (face) from the frame
            face = frame[y:y+h, x:x+w]

            # Detect facial landmarks in the face region
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray_face, dlib.rectangle(0, 0, gray_face.shape[1], gray_face.shape[0]))

            # Detect the facial expression
            emotion_probabilities = emotion_classifier.predict(np.reshape(cv2.resize(gray_face, (64, 64)), (1, 64, 64, 1)) / 255.0)[0]
            emotion_index = np.argmax(emotion_probabilities)
            emotion = emotion_labels[emotion_index]
            probability = emotion_probabilities[emotion_index]

            # Categorize the expressions
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

    # Calculate and print percentages
    print(video_path)
    print("Total frames of the video:", total_frames)
    print("Genuine Smile frames:", genuine_smile_frames, f"({(genuine_smile_frames / total_frames) * 100:.2f}%)")
    print("Fake Smile frames:", fake_smile_frames, f"({(fake_smile_frames / total_frames) * 100:.2f}%)")
    print("Angry frames:", angry_frames, f"({(angry_frames / total_frames) * 100:.2f}%)")
    print("Disgust frames:", disgust_frames, f"({(disgust_frames / total_frames) * 100:.2f}%)")
    print("Fear frames:", fear_frames, f"({(fear_frames / total_frames) * 100:.2f}%)")
    print("Sad frames:", sad_frames, f"({(sad_frames / total_frames) * 100:.2f}%)")
    print("Surprise frames:", surprise_frames, f"({(surprise_frames / total_frames) * 100:.2f}%)")
    print("Neutral frames:", neutral_frames, f"({(neutral_frames / total_frames) * 100:.2f}%)")

    # Release the video capture
    cap.release()