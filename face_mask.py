import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained mask detection model
model = load_model(r"L:\projects\mask_detection\untitled24.py")

# Function to detect and classify face masks in real-time
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_roi, (100, 100))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))
        result = model.predict(reshaped)
        
        label = 'Mask' if result[0][0] > 0.5 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    return frame

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = detect_mask(frame)
    
    cv2.imshow('Face Mask Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
