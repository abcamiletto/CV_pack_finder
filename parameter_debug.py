import cv2 as cv
import gp1_filters as gp
import gp1_utilities
import time

def nothing(x):
    pass

def hsv_window():
    cv.namedWindow("HSV Tracking", cv.WINDOW_NORMAL)
    cv.createTrackbar("add_gaussianblur", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_gauss", "HSV Tracking", 4, 50, nothing)
    cv.createTrackbar("add_medianblur", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_median", "HSV Tracking", 4, 50, nothing)
    cv.createTrackbar("add_bilateralfilter", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_bilateral", "HSV Tracking", 4, 50, nothing)
    cv.createTrackbar("subtraction_or_absdiff", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("add_dilation", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_dilation", "HSV Tracking", 2, 30, nothing)
    cv.createTrackbar("add_erosion", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("kernel_size_erosion", "HSV Tracking", 2, 30, nothing)
    cv.createTrackbar("LV", "HSV Tracking", 50, 255, nothing)
    cv.createTrackbar("UV", "HSV Tracking", 255, 255, nothing)
    cv.createTrackbar("bool_th", "HSV Tracking", 1, 1, nothing)
    cv.createTrackbar("th_value", "HSV Tracking", 50, 255, nothing)
def hsv_parameters():
    l_h = 0
    l_s = 0
    l_v = cv.getTrackbarPos("LV", "HSV Tracking")

    u_h = 255
    u_s = 255
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

    return [l_h, l_s, l_v, u_h, u_s, u_v, bool_gauss, kernel_size_gauss, bool_median, kernel_size_median, bool_bilateral, kernel_size_bilateral, bool_difference, kernel_size_dilation, bool_dilation,kernel_size_erosion, bool_erosion, bool_th, th_value]

def canny_window():
    cv.namedWindow("CannyTracking")
    cv.createTrackbar("low_threshold", "CannyTracking", 15, 255, nothing)
    cv.createTrackbar("high_treshold", "CannyTracking", 100, 255, nothing)
    cv.createTrackbar("iteration", "CannyTracking", 2, 10, nothing)
    cv.createTrackbar("min Area", "CannyTracking", 500, 1000, nothing)
def canny_parameters():
    low = cv.getTrackbarPos("low_threshold", "CannyTracking")
    high = cv.getTrackbarPos("high_treshold", "CannyTracking")
    iteration = cv.getTrackbarPos("iteration", "CannyTracking")
    min_Area = cv.getTrackbarPos("min Area", "CannyTracking")
    return [low, high, iteration, min_Area]



def print_parameters(*lists):
    # print('arguments = ', end="")
    # for list in lists:
    #     if isinstance(list, int):
    #         print(list, end=" ")
    #     else:
    #         for elem in list:
    #             print(elem, end = " ")
    print('filtering_parameters = ' + str(lists[0]))
    print('canny_parameters = ' + str(lists[1]))
    print('resizing_factor = ' + str(resizing_factor))
    output = ''
    for list in lists:
        if isinstance(list, int):
            output += str(list)
            output += ' '
        else:
            for elem in list:
                output += str(elem)
                output += ' '
    return output

def examples_or_camera():
    print('''\nWhich Inputs do you wanna use? Press the corresponding key
        1- Examples
        2- Camera''')
    input1 = int(input())
    if input1 == 1:
        back = cv.imread('images/example_1_b.jpg')
        object = cv.imread('images/example_1_o.jpg')
    if input1 == 2:
        cap = cv.VideoCapture(0);
        print('Press B when you want to choose the background')
        back = gp1_utilities.select_frame(cap)
        print('Press B when you want to choose the objects image')
        object = gp1_utilities.select_frame(cap)
    return back, object

if __name__ == "__main__":
    resizing_factor = 50
    back, object = examples_or_camera()
    back = gp.resize(back, resizing_factor)
    object = gp.resize(object, resizing_factor)
    hsv_window()
    canny_window()
    while True:
        mask, res = gp.filtering(back, object, hsv_parameters())
        canny = gp.canny(res, canny_parameters())
        contour = gp.draw_contour(object, canny)
        cont2, center_list = gp.draw_outer_contours(object,canny, canny_parameters()[-1])

        imgs = [back, object, res, mask, canny, contour, cont2]
        titles = ['Background', 'Packs', "Subtraction", 'Mask', 'Canny', 'Contours', 'Boxed Contours']

        canvas = gp.show_images(imgs, titles)
        cv.namedWindow("Images", cv.WINDOW_NORMAL)
        cv.imshow("Images", canvas)
        cv.resizeWindow('Images', 1400, 800)
        key = cv.waitKey(1)
        if key == 27 & 0xFF:
            break

    string_to_write = print_parameters(hsv_parameters(), canny_parameters(), resizing_factor)
    cv.destroyAllWindows()
    print('\nDo you want to save the parameters values in temp_result.txt? Press 1 if yes, otherwise press 0')
    input3 = int(input())
    if input3 == 1:
        with open('parameters/temp_result.txt', 'w') as file:
            file.write(string_to_write)









