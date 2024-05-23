import cv2 as cv
from cvzone.ClassificationModule import Classifier
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
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
# Load the model and labels
model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

# Load labels from file
with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')


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
    
      # Get predictions
    predict, index = data.getPrediction(img)
        
        # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply thresholding on the gray image to create a binary image
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

        # Find the contours
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        # Take the largest contour based on area
        cnt = max(contours, key=cv.contourArea)
            
            # Compute the bounding rectangle of the contour
        x, y, w, h = cv.boundingRect(cnt)
            
            # Draw the contour and bounding rectangle
            # img = cv.drawContours(img, [cnt], 0, (0, 255, 255), 2)
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
        text = f"Class {class_name}: {max_percentage:.2f}%"
        position = (100, 100)
        cv.putText(img, text, position, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        return {"Nama_Barang": class_name, "Persentase": f"{max_percentage:.2f}%"}
        
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


