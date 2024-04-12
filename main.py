import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import numpy as np

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
    berat = "Direkomendasikan Ke Mobil"
    ringan = "Direkomendasikan Ke Motor"

    def deskripsiringan():
        return {"Nama_Barang": labels[index], "Deskripsi": ringan}

    def deskripsiberat():
        return {"Nama_Barang": labels[index], "Deskripsi": berat}

    def persentase():
        return {"Persentase": f"{round(max(persentasi))}%"} #-> gunakan persentase %
        # return {"Persentase": max(persentasi)} #->gunakan nilai sebenarnya

    descriptions = {
        "Pulpen": deskripsiringan,
        "Kursi": deskripsiberat,
        "Minuman": deskripsiringan,
        "Sofa": deskripsiberat,
        "Makanan": deskripsiringan,
        "Buah": deskripsiringan,
        "Meja": deskripsiberat,
        "Baju": deskripsiringan,
        "Keyboard": deskripsiringan,
        "Kasur": deskripsiberat,
        "Alas Kaki": deskripsiringan,
        "Lemari": deskripsiberat,
        "Tetikus": deskripsiringan,
        "Tas Punggung": deskripsiringan,
        "Laptop": deskripsiberat,
        "Monitor": deskripsiberat,
        "Celana": deskripsiringan,
        "Jaket": deskripsiringan,
        "Buku": deskripsiringan,
    }

    if labels[index] in descriptions:
        obj_desc = descriptions[labels[index]]()
        obj_desc.update(persentase())
        return obj_desc
    else:
        return {"error": "Objek tidak ditemukan dalam deskripsi"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
