import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained MNIST model (you should have this model saved already)
model = tf.keras.models.load_model(r'C:\Users\dhpnr\Desktop\test 1\mnist_model.h5')

# Function to preprocess the image (resize and normalize)
def preprocess_image(thresh):
    # Resize the thresholded image to 28x28
    resized = cv2.resize(thresh, (28, 28))

    # Normalize the pixel values
    normalized = resized.astype('float32') / 255.0

    # Reshape to fit model input (batch_size, height, width, channels)
    reshaped = np.reshape(normalized, (1, 28, 28, 1))

    return reshaped

# Open webcam
cap = cv2.VideoCapture(0)

# Coordinates for the bounding box (adjusted y1 to move the box lower)
x1, y1, width, height = 100, 120, 300, 300  # Moved the y1 down by 50 pixels

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Draw a rectangle for the region of interest (ROI)
    cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)

    # Crop the region inside the rectangle (ROI)
    roi = frame[y1:y1 + height, x1:x1 + width]

    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Threshold the image to make it binary (black and white)
    _, thresh = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)

    # Preprocess the image for prediction
    processed_image = preprocess_image(thresh)

    # Predict the digit using the model
    prediction = model.predict(processed_image)
    predicted_digit = np.argmax(prediction)
    predicted_confidence = np.max(prediction)  # Get the highest probability

    # Display the predicted digit and its confidence (accuracy)
    cv2.putText(frame, f'Prediction: {predicted_digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Accuracy: {predicted_confidence*100:.2f}%', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the predicted digit and accuracy
    cv2.imshow('Webcam - Handwritten Digit Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
