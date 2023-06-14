import cv2
import dlib
from imutils import face_utils
from imutils.video import VideoStream

# Fungsi untuk mendeteksi wajah menggunakan HOG
def detect_faces_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_face_detector = dlib.get_frontal_face_detector()
    faces = hog_face_detector(gray)
    return faces

# Fungsi untuk mendeteksi facial landmarks menggunakan dlib
def detect_landmarks(image, face):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face)
    landmarks = face_utils.shape_to_np(shape)
    return landmarks

# Mengambil gambar dari kamera bawaan komputer atau laptop
cap = VideoStream(src=0).start()

# Menentukan resolusi yang diinginkan (misal: 640x480)
desired_width = 640
desired_height = 480

while True:
    frame = cap.read()

    # Mengubah resolusi gambar menjadi lebih rendah
    frame = cv2.resize(frame, (desired_width, desired_height))

    faces = detect_faces_hog(frame)

    for face in faces:
        landmarks = detect_landmarks(frame, face)

        # Menampilkan kotak persegi panjang pada wajah
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Menampilkan facial landmarks pada wajah
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
