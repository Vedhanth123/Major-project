import cv2
from fer import FER

# Initialize the emotion detector
detector = FER()

# Start webcam feed (usually, 0 is for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break  # Stop if there's an issue with the webcam

    # Analyze emotions in the frame
    result = detector.detect_emotions(frame)

    # Optional: Draw bounding box and emotion label
    for face in result:
        (x, y, w, h) = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        emotion, score = max(face['emotions'].items(), key=lambda item: item[1])
        cv2.putText(frame, f"{emotion} ({score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
