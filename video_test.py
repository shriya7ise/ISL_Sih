import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('check.h5')

# Define preprocessing function
def preprocess_frame(frame):
    # Resize frame to the input size expected by the model
    frame = cv2.resize(frame, (224, 224))  # Replace (224, 224) with your model's input size
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0  # Normalize to [0, 1] range
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Load the video
cap = cv2.VideoCapture('/Users/shriya/Downloads/Screen Recording - Aug 21, 2024.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make prediction
    predictions = model.predict(processed_frame)
    predicted_label = np.argmax(predictions, axis=1)[0]

    # Display the result
    cv2.putText(frame, f'Predicted: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
