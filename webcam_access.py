import cv2 as cv
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

# MODEL ARCHITECTURE

input_layer = Input(shape=(128, 128, 3))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = GlobalAveragePooling2D()(x)

x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)

age_output = Dense(1, name='age')(x)
gender_output = Dense(1, activation='sigmoid', name='gender')(x)

model = Model(
    inputs=input_layer,
    outputs=[age_output, gender_output]
)

# LOAD WEIGHTS

try:
    model.load_weights("models/face_weights.weights.h5")
    print(" Model loaded successfully!")
except Exception as e:
    print(" Model load error:", e)
    exit()

# LOAD FACE DETECTOR

face_cascade = cv.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error loading Haar Cascade!")
    exit()


# OPEN WEBCAM

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam!")
    exit()

print("Webcam started. Press Q to quit.")


# MAIN LOOP


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        cv.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        try:

            # Face Crop
            face = frame[y:y+h, x:x+w]

            # Preprocess
            face_rgb = cv.cvtColor(face, cv.COLOR_BGR2RGB)
            face_resized = cv.resize(face_rgb, (128, 128))

            face_input = face_resized.astype(np.float32) / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            # Prediction
            age_pred, gender_pred = model.predict(
                face_input,
                verbose=0
            )

            # Age
            age = int(round(float(age_pred[0][0])))
            age = max(0, min(age, 100))

            # Gender
            gender_score = float(gender_pred[0][0])

            # UTKFace:
            # 0 = Male
            # 1 = Female
            gender = "Male" if gender_score < 0.5 else "Female"

            # Debug (optional)
            print(
                f"Age={age} | GenderScore={gender_score:.3f}"
            )

            # Display
            cv.putText(
                frame,
                f"Age: {age}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv.putText(
                frame,
                gender,
                (x, y - 35),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Prediction Error:", e)

    cv.imshow("Face Analyser", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv.destroyAllWindows()