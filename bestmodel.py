import cv2 as cv
from cvzone.ClassificationModule import Classifier
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective

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
# Calibration data
known_width_in_pixels = 35  # This should be measured from an image with a known object
known_width_in_cm = 15  # Actual width of the known object in cm
pixel_per_cm = known_width_in_pixels / known_width_in_cm

while True:
    ret, img = cap.read()
    
    if ret:
        # Get predictions
        predict, index = data.getPrediction(img)
        
        # Convert the image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        gray = cv.dilate(gray, None, iterations=2)
        gray = cv.erode(gray, None, iterations=1)

        # Apply thresholding on the gray image to create a binary image
        ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

        # Find the contours
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            # Take the largest contour based on area
            cnt = max(contours, key=cv.contourArea)
            contours = [cnt for cnt in contours if cv.contourArea(cnt) > 500 and cv.arcLength(cnt, True) > 100]
            box = cv.minAreaRect(cnt)  # Corrected to use the largest contour
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            
            # Calculate width and height using the calibrated pixel_per_cm
            wid = euclidean(tl, tr) / pixel_per_cm
            ht = euclidean(tr, br) / pixel_per_cm
            
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            
            # Compute the bounding rectangle of the contour
            x, y, w, h = cv.boundingRect(cnt)
            
            # Draw the contour and bounding rectangle
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
        
        print(f"Class {class_name}: {max_percentage:.2f}%")
        text = f"Class {class_name}: {max_percentage:.2f}% Height: {ht:.2f} cm Width: {wid:.2f} cm"
        position = (100, 100)
        cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Show the image with bounding box
        cv.imshow("Camera", img)

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