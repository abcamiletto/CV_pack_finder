import cv2 as cv
import numpy as np
import copy, math
from imutils import perspective
from imutils import contours
import imutils, datetime

#filtering functions
def filtering(back, object, parameters):
    l_h, l_s, l_v, u_h, u_s, u_v, bool_gauss, kernel_size_gauss, bool_median, kernel_size_median, bool_bilateral, kernel_size_bilateral, bool_difference, kernel_size_dilation, bool_dilation, kernel_size_erosion, bool_erosion, bool_th, th_value = parameters
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
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



    return mask, res
def canny(original, parameters):
    if len(original) > 2:
        original = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    low, high, iteration = parameters

    canny = cv.Canny(original, low, high)
    canny = cv.dilate(canny, None, iterations=iteration)
    canny = cv.erode(canny, None, iterations=iteration)
    return canny

#miscellaneous functions
def resize(img, percentage):
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)
    # resize image
    resized = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    return resized

#contour functions
def draw_contour(img_to_draw, edged):
    img_to_draw_copy = copy.deepcopy(img_to_draw)
    contour, hierarchy = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour = cv.drawContours(img_to_draw_copy, contour, -1, (0, 255, 0), 3)
    return contour
def draw_outer_contours(image, edged):

    height, width, _ = image.shape
    scaling_factor = width / 3000.1

    # find contours in the edge map
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
                            cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    output = image.copy()
    obj_vert = []
    # sort the contours from left-to-right and initialize the bounding box point colors
    if len(cnts) != 0:
        (cnts, _) = contours.sort_contours(cnts)
        colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))
        counter = 0
        # loop over the contours individually
        for (i, c) in enumerate(cnts):
            single_rect_vert = []
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
                single_rect_vert.append([x,y])
            # draw the object num at the top-left corner
            cv.putText(output, "#{}".format(counter + 1),
                        (int(rect[0][0] - 10), int(rect[0][1] - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 3.5 * scaling_factor, (255, 255, 255), round(10 * scaling_factor))
            counter += 1
            obj_vert.append(single_rect_vert)
    center_list = obj_center_calculation(obj_vert)
    return output, center_list
def obj_center_calculation(obj_vert_list):
    center_list = []
    for rect in obj_vert_list:
        center_x = 0
        center_y = 0
        for vert in rect:
            center_x += vert[0]/4
            center_y += vert[1]/4
        center = [center_x, center_y]
        max_dim = round(math.sqrt((center_x - rect[0][0])**2+(center_y-rect[0][1])**2),2)
        center.append(max_dim)
        center.append(datetime.datetime.now())
        center_list.append(center)
    return center_list

#functions used to print different images
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
def grid_needed(n):
    columns = math.ceil(math.sqrt(n))
    if columns * (columns-1) >= n: rows = columns - 1
    else: rows = columns
    return columns, rows