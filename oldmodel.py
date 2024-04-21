import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import numpy as np
from model import descriptions
from imutils import perspective
from imutils import contours as imutils_contours
import imutils
from scipy.spatial.distance import euclidean

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

        predict, index = data.getPrediction(img, color=(255, 0, 0))
        

        confidence_scores = predict


        total_confidence = sum(confidence_scores)


        percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]
        
        edged = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edged = cv.threshold(edged, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        edged = cv.dilate(edged, None, iterations=1)
        edged = cv.erode(edged, None, iterations=1)
        
        contours_img, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours_ref, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        if contours_ref:
            ref_object = max(contours_ref, key=cv.contourArea)
            box = cv.minAreaRect(ref_object)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            dist_in_pixel = euclidean(tl, tr)
            dist_in_cm = 2
            pixel_per_cm = dist_in_pixel / dist_in_cm
        
        for cnt in contours_img:
            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
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
            print(f"Class {class_name}: {percentage:.2f}% Tinggi : {wid} Lebar  :{ht}")


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
