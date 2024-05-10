import cv2 as cv
import numpy as np
from scipy.spatial.distance import euclidean
from imutils import perspective

def box_model(img, pixel_per_cm):
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

            cv.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
            wid = euclidean(tl, tr) / pixel_per_cm
            ht = euclidean(tr, br) / pixel_per_cm
            cv.putText(img, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv.putText(img, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            return img, wid, ht
    return img, None, None