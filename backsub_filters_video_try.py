import cv2 as cv
import numpy as np
import math, copy
from imutils import perspective
from imutils import contours
import imutils

def nothing(x):
    pass
def grid_needed(n):
    columns = math.ceil(math.sqrt(n))
    if columns * (columns-1) >= n: rows = columns - 1
    else: rows = columns
    return columns, rows
def convert_to_3ch(imgs):
    for idx, img in enumerate(imgs):
        if len(img.shape) == 2:
            imgs[idx] = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return imgs
def show_images(imgs, text = None):
    if text and len(imgs) != len(text):
        print('Lenght of lists does not match!')
    imgs = convert_to_3ch(imgs)
    columns, rows = grid_needed(len(imgs))
    height, width, _ = imgs[0].shape
    canvas = np.zeros((height * rows ,width * columns , 3), np.uint8)
    for idx, img in enumerate(imgs):
        row, column = divmod(idx, columns);
        canvas[row * height : (row+1) * height, column * width : (column+1) * width] = img
        if text:
            font = cv.FONT_HERSHEY_SIMPLEX
            canvas = cv.putText(canvas, text[idx], (column * width + round(width/150), (row+1) * height - round(height/50)), font, round(width/800), (50, 100, 255), round(width/500), cv.LINE_AA)
    return canvas
def hsv_filter(back, object,l_h, l_s, l_v, u_h, u_s, u_v, kernel_size_gauss, kernel_size_median, kernel_size_bilateral, k, th_value, kernel_size_erosion):
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    kernel = np.ones((k, k), np.uint8)
    frame = cv.absdiff(object,back)
    frame = cv.GaussianBlur(frame, (kernel_size_gauss, kernel_size_gauss), sigmaX=0, sigmaY=0)  # to remove high frequency noise
    frame = cv.medianBlur(frame, kernel_size_median)  # to remove salt and pepper noise
    frame = cv.bilateralFilter(frame, kernel_size_bilateral, 85, 85)  # to preserve better the edges
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, l_b, u_b)



    _, res = cv.threshold(cv.bitwise_and(frame, frame, mask=mask), th_value, 255, cv.THRESH_TOZERO)
    res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY);
    _, mask = cv.threshold(res_gray, 0, 255, cv.THRESH_BINARY)


    mask = cv.dilate(mask, kernel, iterations=3)
    kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
    mask = cv.erode(mask, kernel_erosion, iterations=3)
    res = cv.bitwise_and(frame, frame, mask=mask)

    return mask, res, frame

def canny_edges(subtraction,low,high):
    canny = cv.Canny(subtraction, low, high)
    canny = cv.dilate(canny, None, iterations=2)
    canny = cv.erode(canny, None, iterations=2)
    return canny
def resize(img, percentage):
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized
def hough_t(img, edges, th):
    img_copy = copy.deepcopy(img)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, th , minLineLength=100, maxLineGap=20)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        result = cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return result
def draw_contours(image, edged):

    height, width, _ = image.shape
    scaling_factor = width / 3000.1

    # find contours in the edge map
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    output = image.copy()
    # sort the contours from left-to-right and initialize the bounding box point colors
    if len(cnts) != 0:
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
        counter = 0
        # loop over the contours individually
        for (i, c) in enumerate(cnts):
            # if the contour is not sufficiently large, ignore it
            if cv.contourArea(c) < 500:
                continue
            # compute the rotated bounding box of the contour, then draw the contours
            box = cv.minAreaRect(c)
            box = cv.boxPoints(box)
            box = np.array(box, dtype="int")
            cv.drawContours(output, [box], -1, (0, 255, 0), round(12 * scaling_factor))

            rect = perspective.order_points(box)
            # loop over the original points and draw them

            for ((x, y), color) in zip(rect, colors):
                cv.circle(output, (int(x), int(y)), round(15 * scaling_factor), color, -1)
            # draw the object num at the top-left corner
            cv.putText(output, "#{}".format(counter + 1),
                        (int(rect[0][0] - 10), int(rect[0][1] - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 3.5 * scaling_factor, (255, 255, 255), round(10 * scaling_factor))
            counter += 1
    return output


if __name__ == "__main__":
    # resizing_factor = 25
    cap = cv.VideoCapture('images/example_1.mp4')
    back = cv.imread('images/example_1_b.jpg')
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (1920,1080))
    if (cap.isOpened() == False):
        print("Error opening Video")
    while True:
        ret, frame = cap.read()
        if ret == True:
            mask, res, subtraction = hsv_filter(back,frame,0,0,51,179,255,255,9,5,5,7,3,3)
            res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
            canny = canny_edges(res_gray, 0, 104)
            cont2 = draw_contours(frame,canny)
            imgs = [back, frame, subtraction, res_gray, mask, canny, cont2]
            titles = ['back', 'frame', 'subtraction', 'res gray', 'mask on subtr', 'Canny', 'cont2']
            canvas = show_images(imgs, titles)
            out.write(cont2)
            cv.namedWindow("video", cv.WINDOW_NORMAL)
            cv.imshow("video", canvas)
            cv.resizeWindow('video', 1800, 1000)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

