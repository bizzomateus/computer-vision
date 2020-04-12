# Find and show the faces from image px-show.jpg

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('images/px-show.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.imread('images/px-show.jpg', 0)
copy = img_rgb.copy()

#HaarCascade cassifier
classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')
faces = classifier.detectMultiScale(img_gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(copy, (x,y), (x+w, y+h), (255, 0, 0), 2)

#showimg
plt.figure(figsize=(10,10))
plt.title("SHOW")
plt.imshow(copy, cmap='gray')

plt.show()