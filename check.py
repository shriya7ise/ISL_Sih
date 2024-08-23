import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('check.h5')

class_labels = ["1", "2", "3", "4", "5", "6", "7", "8"]  # Your Hindi class labels

# Define the coordinates and size of the larger ROI (Region of Interest)
roi_x, roi_y, roi_w, roi_h = 50, 50, 600, 600  # Adjust these values for a larger ROI

def preprocess_image(image):
    image = tf.image.resize(image, [64, 64])
    image = image / 255.0
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

def live_detection():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Define the ROI (Region of Interest) in the frame
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        
        # Convert BGR image to RGB
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Make predictions on the ROI
        predicted_class = predict_image(tf.convert_to_tensor(rgb_roi, dtype=tf.float32))
        
        # Draw a rectangle around the larger ROI
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_class}", (roi_x + 10, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame with the prediction and ROI
        cv2.imshow('Live Detection', frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run live detection
live_detection()
