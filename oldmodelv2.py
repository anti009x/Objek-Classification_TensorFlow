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
    
    # if not ret:
    #     print("Frame captured")
    
    if ret:
        # Get predictions
        predict, index = data.getPrediction(img)
        
        # Convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply thresholding on the gray image to create a binary image
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

        # Find the contours
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take the largest contour based on area
            cnt = max(contours, key=cv.contourArea)
            
            # Compute the bounding rectangle of the contour
            x, y, w, h = cv.boundingRect(cnt)
            
            # Draw the contour and bounding rectangle
            # img = cv.drawContours(img, [cnt], 0, (0, 255, 255), 2)
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Confidence scores for each class
        confidence_scores = predict

        # Sum of all confidence scores
        total_confidence = sum(confidence_scores)

        # Calculate percentage detection for each class
        percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]

        max_index = percentage_detections.index(max(percentage_detections))
        max_percentage = percentage_detections[max_index]
        class_name = labels[max_index]
        
        img, wid, ht = box_model(img, pixel_per_cm)
        persentasi = persentase(persentase_detections)

        print(f"Class {class_name}: {max_percentage:.2f}%")
        text = f"Class {class_name}: {max_percentage:.2f}%"
        position = (100, 100)
        cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

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