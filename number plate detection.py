import cv2

frameWidth = 640
frameHeight = 480
minArea = 500
color = (255, 0, 255)
nPlateCascade = cv2.CascadeClassifier('Resources/haarcascades/haarcascade_russian_plate_number.xml')
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0
while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
            imgRoI = img[y:y+h, x:x+w]
            cv2.imshow("RoI", imgRoI)

    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xff == ord('s'):
        cv2.imwrite('Resources/Scanned/NoPlate_'+str(count)+'.jpg',imgRoI)
        cv2.rectangle(img, (0, 100), (640, 300), (0, 255, 0), -1)
        cv2.putText(img, "Scan saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow('Result', img)
        cv2.waitKey(500)
        count += 1
