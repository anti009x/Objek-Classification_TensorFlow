import cv2 as cv
from cvzone.ClassificationModule import Classifier

# Load the model and labels
model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

# Load labels from file
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

# Video capture setup
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()

    if ret:
        # Get predictions and colorize the bounding box (optional)
        predict, index = data.getPrediction(img, color=(255, 0, 0))
        
        # Confidence scores for each class
        confidence_scores = predict

        # Sum of all confidence scores
        total_confidence = sum(confidence_scores)

        # Calculate percentage detection for each class
        percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]

        # Print percentage detection for each class
        for i, percentage in enumerate(percentage_detections):
            class_name = labels[i]
            print(f"Class {class_name}: {percentage:.2f}%")

        # Show the image with bounding box
        cv.imshow("Kamera", img)

        # Wait for the user to press the ESC key to exit
        key = cv.waitKey(1)
        if key == 27: 
            break
    else:
        print("Invalid frame captured")
        break

# Release resources
cap.release()
cv.destroyAllWindows()
