import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# Capturing Device and Hand Detection
cap = cv2.VideoCapture(2)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# Path saved / Image Saved
path = 'Dataset/Ok'
counter = 0


# Hand Detecting While Statement

while True:
    success, img = cap.read()  # Detector
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Crop Size
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:  # ImgSize
            const = imgSize / h
            wCal = math.ceil(const * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            const = imgSize / w
            hCal = math.ceil(const * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[:, hGap:hCal + hGap] = imgResize

        cv2.imshow("ImgCrop", imgCrop)
        cv2.imshow("imgWhite", imgWhite)

# Saving the image file with the key S to save and q to exit saving at as jpg
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{path}/Image_{time.time()}.jpg', imgWhite) # Save Function
        print(counter)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
