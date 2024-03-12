import cv2 as cv
from cvzone.ClassificationModule import Classifier


model_path = "keras_model.h5" 
labels_path = "labels.txt"  
data = Classifier(model_path, labels_path)

# Initialize video capture from webcam
cap = cv.VideoCapture(0)

while True:

    ret, img = cap.read()


    if ret:

        predict, index = data.getPrediction(img, color=(255, 0, 0))

        # Print prediction and class index for debugging (optional)        print(f"Prediction: {predict}, Class Index: {index}")

        # Draw a rectangle around the detected object (optional)
        # You might need to modify these coordinates based on your model's output
        cv.rectangle(img, (10, 10), (200, 100), (0, 255, 0), 2)

        # Display the predicted class label on the frame (optional)
        cv.putText(img, f"{predict} ({labels_path[index]})", (30, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with predictions
        cv.imshow("frame", img)

        # Handle user input for closing the program
        key = cv.waitKey(1)
        if key == 27:  # Press 'Esc' to quit
            break

    # Exit if frame capture fails
    else:
        print("Error: Frame capture failed")
        break

# Release resources when finished
cap.release()
cv.destroyAllWindows()
