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
known_width_in_pixels = 150  # This should be measured from an image with a known object
known_width_in_cm = 30  # Actual width of the known object in cm
pixel_per_cm = known_width_in_pixels / known_width_in_cm

while True:
    ret, img = cap.read()
    if ret:
        predict, index = data.getPrediction(img, color=(255, 0, 0))
        confidence_scores = predict
        total_confidence = sum(confidence_scores)
        percentage_detections = [(score / total_confidence) * 100 for score in confidence_scores]

        edged = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        edged = cv.threshold(edged, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        edged = cv.dilate(edged, None, iterations=2)
        edged = cv.erode(edged, None, iterations=1)

        contours_img, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours_img:
            contours_img = [cnt for cnt in contours_img if cv.contourArea(cnt) > 500 and cv.arcLength(cnt, True) > 100]
            for contour in contours_img:
                box = cv.minAreaRect(contour)
                box = cv.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box

                # Calculate width and height using the calibrated pixel_per_cm
                wid = euclidean(tl, tr) / pixel_per_cm
                ht = euclidean(tr, br) / pixel_per_cm

                cv.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
                mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
                mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
                cv.putText(img, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv.putText(img, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                for i, percentage in enumerate(percentage_detections):
                    class_name = labels[i]
                    print(f"Class {class_name}: {percentage:.2f}% Tinggi : {ht:.1f}cm Lebar  :{wid:.1f}cm")

        cv.imshow("Kamera", img)
        key = cv.waitKey(1)
        if key == 27:  # ESC key to exit
            break
    else:
        print("Invalid frame captured")
        break

cap.release()
cv.destroyAllWindows()