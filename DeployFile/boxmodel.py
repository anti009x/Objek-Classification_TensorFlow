import cv2 as cv
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective

def box_model(img, pixel_per_cm):
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
            
            # mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            # mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            
            # Compute the bounding rectangle of the contour
            x, y, w, h = cv.boundingRect(cnt)
            
            # Draw the contour and bounding rectangle
            img = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return img, wid, ht
    return img, None, None