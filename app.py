import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import numpy as np
from model import descriptions

app = FastAPI()

model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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

    persentasi = persentase_detections


    def persentase():
        return {"Persentase": f"{round(max(persentasi))}%"} #-> gunakan persentase %
        # return {"Persentase": max(persentasi)} #->gunakan nilai sebenarnya



    if labels[index] in descriptions:
        obj_desc = descriptions[labels[index]]()
        obj_desc.update(persentase())
        return {"Nama_Barang": labels[index], **obj_desc}
    else:
        return {"error": "Objek tidak ditemukan dalam deskripsi"}

