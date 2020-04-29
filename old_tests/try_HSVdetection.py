import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def nothing(x):
    pass

cv.namedWindow("Images", cv.WINDOW_NORMAL)
cv.createTrackbar("LH", "Images", 0, 255, nothing)
cv.createTrackbar("LS", "Images", 0, 255, nothing)
cv.createTrackbar("LV", "Images", 0, 255, nothing)

cv.createTrackbar("UH", "Images", 255, 255, nothing)
cv.createTrackbar("US", "Images", 255, 255, nothing)
cv.createTrackbar("UV", "Images", 255, 255, nothing)

while True:
    frame = cv.imread('../images/image3.jpg')

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    l_h = cv.getTrackbarPos("LH", "Images")
    l_s = cv.getTrackbarPos("LS", "Images")
    l_v = cv.getTrackbarPos("LV", "Images")

    u_h = cv.getTrackbarPos("UH", "Images")
    u_s = cv.getTrackbarPos("US", "Images")
    u_v = cv.getTrackbarPos("UV", "Images")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    mask = cv.inRange(hsv, l_b, u_b)
    mask_3_channel = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    res = cv.bitwise_and(frame, frame, mask=mask)
    img_concate_Hori = np.concatenate((frame, mask_3_channel, res), axis=1)


    cv.imshow("Images", img_concate_Hori)
    cv.resizeWindow('Images', 800, 600)

    # cv.imshow("frame", frame)
    # cv.resizeWindow('frame', 800, 600)
    #
    # cv.namedWindow("mask", cv.WINDOW_NORMAL)
    # cv.imshow("mask", mask)
    # cv.resizeWindow('mask', 800, 600)
    #
    # cv.namedWindow("res", cv.WINDOW_NORMAL)
    # cv.imshow("res", res)
    # cv.resizeWindow('res', 800, 600)


    key = cv.waitKey(1)
    if key == 27:
        break


cv.destroyAllWindows()