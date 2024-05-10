import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import numpy as np
from DeployFile.model import descriptions
from imutils import perspective
from imutils import contours
import imutils
from scipy.spatial.distance import euclidean
from DeployFile.boxmodel import box_model

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
model_path = "DeployFile/keras_model.h5"
labels_path = "DeployFile/labels.txt"
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
  
    known_width_in_pixels = 150.0  # Use a floating point for more precise calculations
    known_width_in_cm = 30.0       # Use a floating point for more precise calculations
    pixel_per_cm = known_width_in_pixels / known_width_in_cm

    labels = [label.strip() for label in labels]
    
    confidence_scores = predict
    total_confidence = sum(confidence_scores)
    persentase_detections = [(score / total_confidence) * 100 for score in confidence_scores]
    
    img, wid, ht = box_model(img, pixel_per_cm)
   
    persentasi = persentase(persentase_detections)

    if labels[index] in descriptions:
        obj_desc = descriptions[labels[index]]()
        obj_desc.update(persentasi)
        return {"Nama_Barang": labels[index], "Lebar_cm": wid, "Tinggi_cm": ht, **obj_desc}
    else:
        return {"error": "Objek tidak ditemukan/ Belum Di Traning"}