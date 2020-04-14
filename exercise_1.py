#crop and save faces from images
import cv2
import matplotlib.pyplot as pyplot

from os import listdir, path, makedirs, getcwd
from os.path import isfile, join

path = 'ex_images'
images_path = [join(path,f) for f in listdir(path) if isfile(join(path, f))]

classifier = cv2.CascadeClassifier('classifiers/haarcascade_frontalface_default.xml')

count = 0
# for every image: open,
for image in images_path:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        img_roi = img[y:y+h, x:x+w]
        img_roi = cv2.resize(img_roi, (162, 162), interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(join(path,'cropped','img_'+str(count)+'.png'), img_roi)
