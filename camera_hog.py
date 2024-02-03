import cv2
import dlib

# Criação do modelo
face_detector = dlib.get_frontal_face_detector()

# Captura da webcam
video_capture = cv2.VideoCapture(0)

while True:
    try:
        ret, frame = video_capture.read()

        # Transforma cada frame em escala de cinza
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detecta as faces
        detections_face = face_detector(frame, 1)

        # Mostra o resultado da detecção em cada frame
        for face in detections_face:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except KeyboardInterrupt:
        break

video_capture.release()
cv2.destroyAllWindows()