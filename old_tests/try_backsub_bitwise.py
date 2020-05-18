import cv2 as cv
import numpy as np

def nothing(x):
    pass
def add_label_to_img(img, text):
    font = cv.FONT_HERSHEY_SIMPLEX
    if len(img.shape) == 2:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img = cv.putText(img, text, (75, 2400), font, 3, (0, 255, 255), 8, cv.LINE_AA)
    return img
def concatenate_img(img1, img2, direction):
    img_conc = np.concatenate((img1, img2), axis=direction)
    return img_conc
def add_with_text(img1, img2, text, direction):
    if direction == 0:
        res = concatenate_img(img1, img2, direction)
        return res
    img2 = add_label_to_img(img2, text)
    res = concatenate_img(img1, img2,direction)
    return res

font = cv.FONT_HERSHEY_SIMPLEX

back = cv.imread('background.jpg')
object = cv.imread('objects.jpg')
back_gray = cv.cvtColor(back, cv.COLOR_BGR2GRAY)
object_gray = cv.cvtColor(object, cv.COLOR_BGR2GRAY)

img1 = cv.subtract(object_gray,back_gray)
img2 = cv.subtract(object,back)


img3 = cv.absdiff(object,back)

gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
## find the nozero regions in the gray
imask =  gray>30
## create a Mat like img2
canvas = np.zeros_like(object, np.uint8)
## set mask
canvas[imask] = object[imask]

row1 = add_with_text(back, object, 'objects', 1)
row2 = add_with_text(add_label_to_img(img2,'subtract'), img3, 'absdiff', 1 )
result = add_with_text(row1, row2, '', 0)

cv.namedWindow("img1", cv.WINDOW_NORMAL)
cv.imshow("img1", result)
cv.resizeWindow('img1', 1800, 1000)

cv.namedWindow("Images", cv.WINDOW_NORMAL)
cv.createTrackbar("add_gaussianblur", "Images", 0, 1, nothing)
cv.createTrackbar("add_medianblur", "Images", 0, 1, nothing)
cv.createTrackbar("add_bilateralfilter", "Images", 0, 1, nothing)
cv.createTrackbar("LH", "Images", 0, 255, nothing)
cv.createTrackbar("LS", "Images", 0, 255, nothing)
cv.createTrackbar("LV", "Images", 0, 255, nothing)

cv.createTrackbar("UH", "Images", 255, 255, nothing)
cv.createTrackbar("US", "Images", 255, 255, nothing)
cv.createTrackbar("UV", "Images", 255, 255, nothing)

while True:


    l_h = cv.getTrackbarPos("LH", "Images")
    l_s = cv.getTrackbarPos("LS", "Images")
    l_v = cv.getTrackbarPos("LV", "Images")

    u_h = cv.getTrackbarPos("UH", "Images")
    u_s = cv.getTrackbarPos("US", "Images")
    u_v = cv.getTrackbarPos("UV", "Images")

    bool_gauss = cv.getTrackbarPos("add_gaussianblur", 'Images')
    bool_median = cv.getTrackbarPos("add_medialblur", 'Images')
    bool_bilateral = cv.getTrackbarPos("add_bilateralfilter", 'Images')

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    frame = img2

    if bool_gauss == 1:
        frame = cv.GaussianBlur(frame, (35, 35), 15)  # to remove high frequency noise
    if bool_median == 1:
        frame = cv.medianBlur(frame, 35)  # to remove salt and pepper noise
    if bool_bilateral == 1:
        frame = cv.bilateralFilter(frame, 15, 125, 125)  # to preserve better the edges

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, l_b, u_b)
    mask_3_channel = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    res = cv.bitwise_and(frame, frame, mask=mask)
    img_concate_Hori = np.concatenate((frame, mask_3_channel, res), axis=1)


    cv.imshow("Images", img_concate_Hori)
    cv.resizeWindow('Images', 1800, 1000)

    key = cv.waitKey(1)
    if key == 27:
        break


cv.destroyAllWindows()


# cv.waitKey(0)
# cv.destroyAllWindows()