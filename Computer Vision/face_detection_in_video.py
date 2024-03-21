import cv2 as cv

def draw_rect(img):
    l=30
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255,0,255), thickness=1)
        # topleft
        cv.line(img, (x, y), (x + l, y), (255,0,255), 3)
        cv.line(img, (x, y), (x, y + l), (255,0,255), 3)
        # topright
        cv.line(img, (x + w, y), (x + w - l, y), (255,0,255), 3)
        cv.line(img, (x + w, y), (x + w , y + l), (255,0,255), 3)
        # bottomleft
        cv.line(img, (x, y + h - l), (x , y + h), (255,0,255), 3)
        cv.line(img, (x, y + h), (x + l, y + h), (255,0,255), 3)
        # bottomright
        cv.line(img, (x + w - l, y + h), (x + w, y + h), (255,0,255), 3)
        cv.line(img, (x + w, y + h - l), (x + w, y + h), (255,0,255), 3)

haar_cascade = cv.CascadeClassifier('haar_cascade.xml')

cap = cv.VideoCapture(0)

while True:
    f_,img= cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces=haar_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 3)
    draw_rect(img)

    cv.imshow('Live Detection', img)
    k = cv.waitKey(100) & 0xFF
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()