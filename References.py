import cv2
import numpy as np

# Read images

'''
img = cv2.imread('Resources/lenna_opencv.png')
cv2.imshow("Image", img)
cv2.waitKey(0)
'''


# Read videos

'''
cap = cv2.VideoCapture('Resources/Road.mp4')

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(50) & 0xff == ord('q'):
        break
'''


# Using webcam

'''
cap = cv2.VideoCapture(0)
cap.set(3, 640)     # width (id = 3) = 640
cap.set(4, 480)     # height (id = 4) = 480
cap.set(10, 100)    # brightness (id = 10) = 100

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(50) & 0xff == ord('q'):
        break
'''


# Colors, blur, canny, dilation, erosion

'''
kernel = np.ones((5, 5), np.uint8)              # 5x5 matrix of 1, np.uint8 --> values range from 0 to 255
img = cv2.imread('Resources/lenna_opencv.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)      # (7, 7) --> kernel (has to be odd), 0 --> sigmax
imgCanny = cv2.Canny(img ,100, 100)             # 100, 100 --> threshold values
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
imgEroded = cv2.erode(imgDilation, kernel, iterations = 1)

cv2.imshow("Orig img", img)
cv2.imshow("Gray img", imgGray)
cv2.imshow("Blur img", imgBlur)
cv2.imshow("Canny img", imgCanny)
cv2.imshow("Dilation img", imgDilation)
cv2.imshow("Eroded img", imgEroded)

cv2.waitKey(0)
'''


# Resizing and cropping

'''
# Origin of an image is top left corner

img = cv2.imread('Resources/lambo.png')
img = cv2.resize(img, (623, 462))              # (width, height)
imgResize = cv2.resize(img, (1000, 500))
print(img.shape)                               # gives size
print(imgResize.shape)                         # gives size
imgCrop = img[0:200, 200:500]                  # (height, width)

cv2.imshow("Original", img)
cv2.imshow("Cropped", imgCrop)
cv2.imshow("Resize", imgResize)

cv2.waitKey(0)
'''


# Adding shapes and text

'''
# line
img = np.zeros((512, 512, 3), np.uint8)
img2 = np.zeros((512, 512, 3), np.uint8)
img3 = np.zeros((512, 512, 3), np.uint8)
img[200:300, 100:300] = 255, 0, 0                                                   # img[:] = 255, 0, 0 for full blue
cv2.line(img2, (0, 0), (300, 100), (0, 255, 0), 5)                                  # (image, (starting x, y), (ending x, y), (color), thickness)
cv2.line(img3, (0, 0), (img3.shape[1], img3.shape[0]), (0, 255, 0), 5)              # (image, (starting x, y), (full width, full height), (color), thickness)

cv2.imshow("img", img)
cv2.imshow("img2", img2)
cv2.imshow("img3", img3)


# rectangle
img = np.zeros((512, 512, 3), np.uint8)
img2 = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, (100, 200), (300, 500), (0, 255, 255), 3)                        # (image, (starting x, y), (opposite corner x, y), ,.. )
cv2.rectangle(img2, (0, 0), (100, 200), (0, 0, 255), cv2.FILLED)                    # filled rectangle (cv2.FILLED is same as -1)

cv2.imshow("Rectangle", img)
cv2.imshow("Filled rectangle", img2)


# circle
img = np.zeros((512, 512, 3), np.uint8)
cv2.circle(img, (400, 50), 30, (255, 255, 0), 2)                                    # (image, (center x, y), radius, ..)

cv2.imshow("Circle", img)


# Text
img = np.zeros((512, 512, 3), np.uint8)
cv2.putText(img, "OpenCV", (256, 256), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 150, 0), 1)  # (image, text, origin, font, scale (float), color, thickness)

cv2.imshow("Text", img)



cv2.waitKey(0)
'''


# Warp image

'''
img = cv2.imread("Resources/cards.jpg")                                     
width, height = 250, 350
pts1 = np.float32([ [111, 219], [287, 188], [154, 482], [352, 440] ])       # float array with 4 corners of cards
pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])        # defining which corner is what
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgOutput = cv2.warpPerspective(img, matrix, (width, height))

cv2.imshow("title", img)
cv2.imshow("output", imgOutput)
cv2.waitKey(0)
'''


# Joining images

'''
img = cv2.imread('Resources/lenna_opencv.png')

hor = np.hstack((img, img))
ver = np.vstack((img, img))

_2x2 = np.vstack((hor, hor))

cv2.imshow("2 x 2", _2x2)
cv2.imshow("hstack", hor)
cv2.imshow("vstack", ver)

cv2.waitKey(0)
'''


# Color detection

'''
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread('Resources/lambo.png')
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    /*
    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    */
    imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
    cv2.imshow("Stacked", imgStack)
    cv2.waitKey(0)
'''


# Stacking images

'''
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


imgStack1 = stackImages(0.6, ([img1, img2], [img3, img4]))
imgStack2 = stackImages(0.6, ([img1, img2, img3, img4]))

cv2.imshow("Stacked1", imgStack1)
cv2.imshow("Stacked2", imgStack2)

cv2.waitKey(0)
'''


# color & contour detection

'''
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:                                                                          # 500 pixels
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)                               # -1 for all
            peri = cv2.arcLength(cnt, True)                                                     # True <-- closed
            print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objCor = len(approx)
            x, y, width, height = cv2.boundingRect(approx)
            if objCor == 3:
                objectType = 'Triangle'
            elif objCor == 4:
                aspRatio = width/float(height)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objectType = 'Square'
                else:
                    objectType = 'Rectangle'
            elif objCor > 4:
                objectType = 'Circle'
            else:
                objectType = 'None'
            cv2.rectangle(imgContour, (x, y), (x+width, y+height), (0, 255, 0), 2)              # can find center, total height, width
            cv2.putText(imgContour, objectType, (x+(width//2), y+(height//2)), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)


img = cv2.imread('Resources/shapes.png')
imgContour = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgBlack = np.zeros_like(img)
getContours(imgCanny)
imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlack]))
cv2.imshow("Stack", imgStack)

cv2.waitKey(0)
'''


# Face detection

'''
faceCascade = cv2.CascadeClassifier('Resources/haarcascades/haarcascade_frontalface_default.xml')
img = cv2.imread('Resources/lenna_opencv.png')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
cv2.imshow("Result", img)
cv2.waitKey(0)
'''

# Video face detection


faceCascade = cv2.CascadeClassifier('Resources/haarcascades/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "some name", (x+w//2, y-10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

