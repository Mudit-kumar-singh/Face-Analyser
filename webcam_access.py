import cv2 as cv

face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("could not read frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    detect_face = face_cascade.detectMultiScale(gray, 1.1, 5)

    for(x,y,w,h) in detect_face:
        cv.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)

    cv.imshow("webcam face detection", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        print("quit")
        break
cap.release()
cv.destroyAllWindows()