import cv2
import face_recognition
from fer import FER
import os

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])

# Initialize camera
cap = cv2.VideoCapture(0)

# Emotion detector
emotion_detector = FER(mtcnn=True)

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect face locations and encodings on small frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if True in matches:
            best_match_index = face_distances.argmin()
            name = known_face_names[best_match_index]

        # Scale face coordinates back to original frame
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Emotion detection on full-resolution face region
        face_crop = frame[top:bottom, left:right]
        emotion_result = emotion_detector.detect_emotions(face_crop)
        emotion = ""
        if emotion_result:
            emotion = max(emotion_result[0]["emotions"], key=emotion_result[0]["emotions"].get)

        # Draw rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{name} | {emotion}" if emotion else name
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face & Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
