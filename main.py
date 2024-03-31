import cv2 as cv
from cvzone.ClassificationModule import Classifier

model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, img = cap.read()

    if ret:
        predict, index = data.getPrediction(img, color=(255, 0, 0))

        with open(labels_path, 'r') as file:
            labels = file.readlines()

        labels = [label.strip() for label in labels]
        #Baru 1 Model   
        if labels[index] == "Pulpen":
            deskripsi = "Barang Direkomendasikan Motor"
            persentase = "100%"
            nama = "Pulpen"
            klasifikasi = "Ringan"
            
            print("Nama:", labels[index])
            print("Deskripsi:", deskripsi)
            print("Persentase Objek Detection:", persentase)
            print("Klasifikasi:", klasifikasi)

        cv.imshow("Kamera", img)

        key = cv.waitKey(1)
        if key == 27: 
            break
    else:
        print("Invalid")
        break

cap.release()
cv.destroyAllWindows()
