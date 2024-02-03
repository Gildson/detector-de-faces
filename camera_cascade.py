import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyes_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

video_capture = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video_capture.read()

        # Transforma cada frame em escala de cinza
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta as faces
        detections_face = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                    minNeighbors=5)

        # Detecta os olhos
        detections_eyes = eyes_detector.detectMultiScale(image_gray, minNeighbors=5)

        # Mostra o resultado da detecção em cada frame
        for (x, y, w, h) in detections_face:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        for (x_e, y_e, w_e, h_e) in detections_eyes:
            cv2.rectangle(frame, (x_e, y_e), (x_e+w_e, y_e+h_e), (255, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except KeyboardInterrupt:
        break

video_capture.release()
cv2.destroyAllWindows()