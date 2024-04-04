import cv2

def detect_faces(frame, cascade_frontal, cascade_profile):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_frontal = cascade_frontal.detectMultiScale(gray, 1.3, 5)
    faces_profile = cascade_profile.detectMultiScale(gray, 1.3, 5)

    # Selección del rostro con mayor área
    max_area_frontal = 0
    max_area_profile = 0
    max_face_frontal = None
    max_face_profile = None

    for (x, y, w, h) in faces_frontal:
        area = w * h
        if area > max_area_frontal:
            max_area_frontal = area
            max_face_frontal = (x, y, w, h)

    for (x, y, w, h) in faces_profile:
        area = w * h
        if area > max_area_profile:
            max_area_profile = area
            max_face_profile = (x, y, w, h)

    # Devuelve solo el rostro con mayor área
    return max_face_frontal, max_face_profile

def detect_wink(frame, cascade_eye):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = cascade_eye.detectMultiScale(gray, 1.3, 5)
    return len(eyes) == 1

def main():
    cascade_frontal_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade_profile_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
    face_cascade_frontal = cv2.CascadeClassifier(cascade_frontal_path)
    face_cascade_profile = cv2.CascadeClassifier(cascade_profile_path)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_frontal, face_profile = detect_faces(frame, face_cascade_frontal, face_cascade_profile)

        if face_frontal is not None:
            x, y, w, h = face_frontal
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Detectar guiño solo si se detecta una cara frontal
            if detect_wink(frame, eye_cascade):
                cv2.putText(frame, "Wink Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
        elif face_profile is not None:
            x, y, w, h = face_profile
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
