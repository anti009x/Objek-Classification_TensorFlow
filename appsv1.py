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
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

# Calibration data
known_width_in_pixels = 150 
known_width_in_cm = 30 
pixel_per_cm = known_width_in_pixels / known_width_in_cm

while True:
    ret, img = cap.read()
    if ret:
        predict, index = data.getPrediction(img, color=(255, 0, 0))
        confidence_scores = predict
        total_confidence = sum(confidence_scores)
        percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]

    
        print(f"Predictions: {predict}")
        print(f"Indices: {index}")

        if isinstance(index, np.int64):
            index = [int(index)]
        elif isinstance(index, list):
            print("Multiple detections:", index)

        predicted_classes = [labels[i] for i in index]  


        class_counts = {label: predicted_classes.count(label) for label in labels}
        
        # Display the count of each detected class on the image
        y_offset = 50
        for label, count in class_counts.items():
            cv.putText(img, f'{label}: {count}', (50, y_offset), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            y_offset += 100  
        
        edged = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edged = cv.threshold(edged, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        edged = cv.dilate(edged, None, iterations=2)  
        edged = cv.erode(edged, None, iterations=1)

        contours_img, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours_img:
          
            contours_img = [cnt for cnt in contours_img if cv.contourArea(cnt) > 500 and cv.arcLength(cnt, True) > 100]
            if contours_img:
                ref_object = max(contours_img, key=cv.contourArea)
                box = cv.minAreaRect(ref_object)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box

                cv.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)  # Draw the largest contour in green
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
                mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
                wid = euclidean(tl, tr) / pixel_per_cm
                ht = euclidean(tr, br) / pixel_per_cm
                cv.putText(img, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv.putText(img, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                for i, percentage in enumerate(percentage_detections):
                    class_name = labels[i]
                    class_count = class_counts[class_name]
                    print(f"Class {class_name}: {percentage:.2f}% Height: {wid:.1f}cm Width: {ht:.1f}cm Count: {class_count}")
                    
        cv.imshow("Kamera", img)
        key = cv.waitKey(1)
        if key == 27:
            break
    else:
        print("Invalid frame captured")
        break

# Release resources
cap.release()
cv.destroyAllWindows()
