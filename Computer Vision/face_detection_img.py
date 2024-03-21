import cv2 as cv
import numpy as np

img = cv.imread('safl pors.jpeg')
nimg= cv.resize(img,(1000,700),interpolation=cv.INTER_CUBIC)
cv.imshow('pors',nimg)

#gray=cv.cvtColor(nimg, cv.COLOR_BGR2GRAY, )
#cv.imshow('Gray',gray)

haar_cascade=cv.CascadeClassifier('haar_faces.xml')
faces_rect=haar_cascade.detectMultiScale(nimg,scaleFactor=1.1,minNeighbors=1)
print(f'Numeber of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(nimg, (x,y), (x+w,y+h), (255,0,255), thickness=2)

cv.imshow('Faces',nimg)

cv.waitKey(0)
