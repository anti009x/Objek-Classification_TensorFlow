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

        persentasi = "100%"
        berat= "Direkomendasikan Ke Motor"
        ringan= "Direkomendasikan Ke Motor"


        def deskripsiringan():
            print("Nama:", labels[index])
            print (f"Deskripsi : {berat}") 
            

        def deskripsiberat():
            print("Nama:", labels[index])
            print (f"Deskripsi : {berat}") 

        def persentase():
            print (f"Persentase Dectection Objek :  {persentasi}") 


        if labels[index] == "Pulpen":
            deskripsiringan()
            persentase()

        elif labels[index] == "Kursi":
            deskripsiringan()
            persentase()
        
        elif labels[index] == "Minuman":
            deskripsiringan()
            persentase()

        elif labels[index] == "Sofa":
            deskripsiberat()
            persentase()

        elif labels[index] == "Makanan":
            deskripsiringan()
            persentase()

        elif labels[index] == "Buah":
            deskripsiringan()
            persentase()

        elif labels[index] == "Meja":
            deskripsiberat()
            persentase()

        elif labels[index] == "Baju":
            deskripsiringan()
            persentase()

        elif labels[index] == "Keyboard":
            deskripsiringan()
            persentase()

        elif labels[index] == "Kasur":
            deskripsiberat()
            persentase()

        elif labels[index] == "Alas Kaki":
            deskripsiringan() 
            persentase()

        elif labels[index] == "Lemari":
            deskripsiberat()
            persentase()

        elif labels[index] == "Tetikus":
            deskripsiringan()
            persentase()

        elif labels[index] == "Tas Punggung":
            deskripsiringan()
            persentase()

        elif labels[index] == "Laptop":
            deskripsiberat()
            persentase()

        elif labels[index] == "Monitor":
            deskripsiberat()
            persentase()

        elif labels[index] == "Celana":
            deskripsiringan()
            persentase()

        elif labels[index] == "Jaket":
            deskripsiringan()
            persentase()

        elif labels[index] == "Buku":
            deskripsiringan()
            persentase()

        cv.imshow("Kamera", img)

        key = cv.waitKey(1)
        if key == 27: 
            break
    else:
        print("Invalid")
        break

cap.release()
cv.destroyAllWindows()
