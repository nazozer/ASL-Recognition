import cv2
import numpy as np
import math
import tensorflow as tf
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

offset = 20
imgSize = 300
word = ""
last_prediction = None

with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split()[1] for line in f.readlines()]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        imgInput = cv2.resize(imgWhite, (224, 224))  # Modelin giriş boyutuna göre resize
        imgInput = (imgInput.astype(np.float32) / 255.0)
        imgInput = np.expand_dims(imgInput, axis=0)

        interpreter.set_tensor(input_details[0]['index'], imgInput)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction = np.squeeze(output_data)
        index = np.argmax(prediction)

        last_prediction = labels[index]

        # Çizimler
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, last_prediction, (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)

    # El yoksa, son harfi ekranda tut
    if not hands and last_prediction:
        cv2.putText(imgOutput, last_prediction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Her döngüde tuşa bak
    key = cv2.waitKey(1)

    if key == 27:  # ESC tuşuna basılırsa
        print("Exiting the program...")
        break

    if key == 13 and last_prediction:  # Enter tuşu
        word += last_prediction
        print(f"Added letter: {last_prediction}")

    elif key == 32:  # Space tuşu
        word += ' '
        print("Added space")

    elif key == 8:  # Backspace tuşu
        word = word[:-1]
        print("Deleted last character")

    # Ekrana kelimeyi yaz
    cv2.putText(imgOutput, "Word: " + word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    cv2.imshow("Image", imgOutput)

