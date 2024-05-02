import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import numpy as np
from model import descriptions
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial.distance import euclidean

from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

app = FastAPI(docs_url=None, redoc_url=None)
# app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://unpkg.com/swagger-ui-dist@5.9.0/swagger-ui.css",
    )
@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://unpkg.com/redoc@next/bundles/redoc.standalone.js",
    )

async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()
model_path = "keras_model.h5"
labels_path = "labels.txt"
data = Classifier(model_path, labels_path)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

def persentase(persentasi):
    return {"Persentase": f"{round(max(persentasi))}%"}

@app.get("/")
async def read_root():
    return {"API Sedang Jalan"}

@app.post("/uploadgambar/")
async def klasifikasi(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    predict, index = data.getPrediction(img, color=(255, 0, 0))

    with open(labels_path, 'r') as file:
        labels = file.readlines()

    labels = [label.strip() for label in labels]
    
    confidence_scores = predict
    total_confidence = sum(confidence_scores)
    persentase_detections = [(score / total_confidence) * 100 for score in confidence_scores]
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)

    edged = cv.Canny(blur, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)
    
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    cnts = [x for x in cnts if cv.contourArea(x) > 100]
    # cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]



    ref_object = cnts[0]
    box = cv.minAreaRect(ref_object)
    box = cv.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    dist_in_cm = 2
    pixel_per_cm = dist_in_pixel/dist_in_cm
        
    for cnt in cnts:
            box = cv.minAreaRect(cnt)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            cv.drawContours(img, [box.astype("int")], -1, (0, 0, 255), 2)
            mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
            mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
            wid = euclidean(tl, tr)/pixel_per_cm
            ht = euclidean(tr, br)/pixel_per_cm
            cv.putText(img, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            cv.putText(img, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), 
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    persentasi = persentase(persentase_detections)

    if labels[index] in descriptions:
        obj_desc = descriptions[labels[index]]()
        obj_desc.update(persentasi)
        return {"Nama_Barang": labels[index], "Lebar_cm": wid, "Tinggi_cm": ht, **obj_desc}
    else:
        return {"error": "Objek tidak ditemukan/ Belum Di Traning"}