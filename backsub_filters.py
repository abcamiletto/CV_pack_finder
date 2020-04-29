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
def hsv_window():
    cv.namedWindow("HSV Tracking", cv.WINDOW_NORMAL)
    cv.createTrackbar("add_gaussianblur", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_gauss", "HSV Tracking", 5, 50, nothing)
    cv.createTrackbar("add_medianblur", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_median", "HSV Tracking", 5, 50, nothing)
    cv.createTrackbar("add_bilateralfilter", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_bilateral", "HSV Tracking", 5, 50, nothing)
    cv.createTrackbar("subtraction_or_absdiff", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("add_dilation", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_dilation", "HSV Tracking", 2, 30, nothing)
    cv.createTrackbar("add_erosion", "HSV Tracking", 0, 1, nothing)
    cv.createTrackbar("kernel_size_erosion", "HSV Tracking", 2, 30, nothing)
    cv.createTrackbar("LH", "HSV Tracking", 0, 179, nothing)
    cv.createTrackbar("LS", "HSV Tracking", 0, 255, nothing)
    cv.createTrackbar("LV", "HSV Tracking", 55, 255, nothing)
    cv.createTrackbar("UH", "HSV Tracking", 179, 179, nothing)
    cv.createTrackbar("US", "HSV Tracking", 255, 255, nothing)
    cv.createTrackbar("UV", "HSV Tracking", 255, 255, nothing)
    cv.createTrackbar("bool_th", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("th_value", "HSV Tracking", 50, 255, nothing)
def hsv_parameters():
    l_h = cv.getTrackbarPos("LH", "HSV Tracking")
    l_s = cv.getTrackbarPos("LS", "HSV Tracking")
    l_v = cv.getTrackbarPos("LV", "HSV Tracking")

    u_h = cv.getTrackbarPos("UH", "HSV Tracking")
    u_s = cv.getTrackbarPos("US", "HSV Tracking")
    u_v = cv.getTrackbarPos("UV", "HSV Tracking")

    bool_gauss = cv.getTrackbarPos("add_gaussianblur", 'HSV Tracking')
    kernel_size_gauss = cv.getTrackbarPos("kernel_size_gauss", 'HSV Tracking')
    if kernel_size_gauss % 2 == 0:
        kernel_size_gauss += 1
    bool_median = cv.getTrackbarPos("add_medianblur", 'HSV Tracking')
    kernel_size_median = cv.getTrackbarPos("kernel_size_median", 'HSV Tracking')
    if kernel_size_median % 2 == 0:
        kernel_size_median += 1
    bool_bilateral = cv.getTrackbarPos("add_bilateralfilter", 'HSV Tracking')
    kernel_size_bilateral = cv.getTrackbarPos("kernel_size_bilateral", 'HSV Tracking')
    if kernel_size_bilateral % 2 == 0:
        kernel_size_bilateral += 1
    bool_difference = cv.getTrackbarPos("subtraction_or_absdiff", 'HSV Tracking')
    bool_dilation = cv.getTrackbarPos("add_dilation", 'HSV Tracking')
    kernel_size_dilation = cv.getTrackbarPos("kernel_size_dilation", 'HSV Tracking')
    bool_erosion = cv.getTrackbarPos("add_erosion", 'HSV Tracking')
    kernel_size_erosion = cv.getTrackbarPos("kernel_size_erosion", 'HSV Tracking')
    bool_th = cv.getTrackbarPos("bool_th", 'HSV Tracking')
    th_value = cv.getTrackbarPos("th_value", 'HSV Tracking')

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    return l_b, u_b, bool_gauss, kernel_size_gauss, bool_median, kernel_size_median, bool_bilateral, kernel_size_bilateral, bool_difference, kernel_size_dilation, bool_dilation,kernel_size_erosion, bool_erosion, bool_th, th_value
def hsv_filter(back, object):
    l_b, u_b, bool_gauss, kernel_size_gauss, bool_median, kernel_size_median, bool_bilateral, kernel_size_bilateral, bool_difference, kernel_size_dilation, bool_dilation, kernel_size_erosion, bool_erosion, bool_th, th_value = hsv_parameters()

    if not bool_difference:
        frame = cv.subtract(object,back)
    else:
        frame = cv.absdiff(object,back)
    if bool_gauss:
        frame = cv.GaussianBlur(frame, (kernel_size_gauss, kernel_size_gauss), sigmaX=0, sigmaY=0)  # to remove high frequency noise
    if bool_median:
        frame = cv.medianBlur(frame, kernel_size_median)  # to remove salt and pepper noise
    if bool_bilateral:
        frame = cv.bilateralFilter(frame, kernel_size_bilateral, 85, 85)  # to preserve better the edges

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, l_b, u_b)


    if bool_th:
        _, res = cv.threshold(cv.bitwise_and(frame, frame, mask=mask), th_value, 255, cv.THRESH_TOZERO)
        res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY);
        _, mask = cv.threshold(res_gray, 0, 255, cv.THRESH_BINARY)

    if bool_dilation:
        kernel_dilation = np.ones((kernel_size_dilation, kernel_size_dilation), np.uint8)
        mask = cv.dilate(mask, kernel_dilation, iterations=3)
    res = cv.bitwise_and(frame, frame, mask=mask)

    if bool_erosion:
        kernel_erosion = np.ones((kernel_size_erosion, kernel_size_erosion), np.uint8)
        mask = cv.erode(mask, kernel_erosion, iterations=3)
    res = cv.bitwise_and(frame, frame, mask=mask)



    return mask, res, frame


def canny_window():
    cv.namedWindow("CannyTracking")
    cv.createTrackbar("low_threshold", "CannyTracking", 15, 255, nothing)
    cv.createTrackbar("high_treshold", "CannyTracking", 80, 255, nothing)
    cv.createTrackbar("subtract_or_original", "CannyTracking", 0, 1, nothing)
    cv.createTrackbar("blurring_original", "CannyTracking", 0, 1, nothing)
def canny_parameters():
    low = cv.getTrackbarPos("low_threshold", "CannyTracking")
    high = cv.getTrackbarPos("high_treshold", "CannyTracking")
    bool_original = cv.getTrackbarPos("subtract_or_original", "CannyTracking")
    original_blur = cv.getTrackbarPos("blurring_original", "CannyTracking")
    return low, high, bool_original, original_blur
def canny_edges(subtraction, original):
    low, high, bool_original, original_blur = canny_parameters()
    if bool_original:
        if original_blur:
            original = cv.GaussianBlur(original, (3,3), 3)
        canny = cv.Canny(original, low, high)
    else:
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
            if cv.contourArea(c) < 100:
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
    resizing_factor = 100
    back = resize(cv.imread('images/example_1_b.jpg'), resizing_factor)
    object = resize(cv.imread('images/example_1_o.jpg'), resizing_factor)
    # subtraction = cv.subtract(object,back)
    hsv_window()
    # th_window()
    # th_adaptive_window()
    canny_window()
    while True:
        mask, res, subtraction = hsv_filter(back, object)
        res_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        # th1 = th_filter(subtraction)
        # mean_th, gauss_th = th_adaptive_filter(subtraction)

        canny = canny_edges(res_gray, object)


        object_copy = copy.deepcopy(object)
        contour, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contour = cv.drawContours(object_copy, contour, -1, (0, 255, 0), 3)

        cont2 = draw_contours(object,canny)

        # hough = hough_t(object, canny, 10)

        imgs = [back, object, subtraction,  res_gray, mask, canny, contour, cont2]
        titles = ['back', 'object', 'subtraction', 'res gray','mask on subtr', 'Canny', 'contour', 'cont2']

        canvas = show_images(imgs, titles)

        cv.namedWindow("Images", cv.WINDOW_NORMAL)
        cv.imshow("Images", canvas)
        cv.resizeWindow('Images', 1800, 1000)

        key = cv.waitKey(2)
        if key == 27 & 0xFF:
            break


    cv.destroyAllWindows()

