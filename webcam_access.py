import cv2 as cv
import numpy as np
import tensorflow as tf

try:
    model = tf.keras.models.load_model(
        "models/face_model.keras",
        compile=False
    )
    print("Model loaded successfully!")
except Exception as e:
    print("Model load error:", e)
    exit()

face_cascade = cv.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Error loading Haar Cascade")
    exit()

cap = cv.VideoCapture(0, cv.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam started. Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        try:

            face_rgb = cv.cvtColor(
                face,
                cv.COLOR_BGR2RGB
            )

            face_resized = cv.resize(
                face_rgb,
                (128, 128)
            )

            face_input = (
                face_resized.astype(np.float32)
                / 255.0
            )

            face_input = np.expand_dims(
                face_input,
                axis=0
            )

            age_pred, gender_pred = model.predict(
                face_input,
                verbose=0
            )

            age = int(
                round(
                    float(age_pred.flatten()[0])
                )
            )

            age = max(0, min(age, 100))

            gender_score = float(
                gender_pred.flatten()[0]
            )

            gender = (
                "Male"
                if gender_score < 0.5
                else "Female"
            )

            cv.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv.putText(
                frame,
                f"{gender}, {age}",
                (x, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        except Exception as e:
            print("Prediction error:", e)

    cv.imshow("Face Analyser", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()