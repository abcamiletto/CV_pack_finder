import cv2 as cv
import gp1_filters as gp
import time
import gp1_utilities as gp_out

def example_or_camera():
    print('''Do you wanna use the camera? Press the corresponding key
        1- No, use example_1.mp4 and its background
        2- Yes, use the camera''')
    input1 = int(input())
    if input1 == 1:
        cap = cv.VideoCapture('images/example_1.mp4')
        back = cv.imread('images/example_1_b.jpg')
    elif input1 == 2:
        cap = cv.VideoCapture(0);
        print('\nPress B when you want to choose the background')
        back = gp_out.select_frame(cap)
    else:
        print('Wrong Value')
    return cap, back

def get_parameters():
    print('''\nWhich parameters do you wanna use? Press the corresponding key
    1- Default parameters
    2- Last parameters saved from parameter_debug.py''')
    input2 = int(input())
    if input2 == 1:
        with open('parameters/default.txt', 'r') as file:
            input_from_file = file.readline().rstrip(' ')
        input_list = [int(i) for i in input_from_file.split(" ")]
        filtering_parameters = input_list[:19]
        canny_parameters = input_list[19:23]
        resizing_factor = input_list[-1]
    elif input2 == 2:
        with open('parameters/temp_result.txt', 'r') as file:
            input_from_file = file.readline().rstrip(' ')
        input_list = [int(i) for i in input_from_file.split(" ")]
        filtering_parameters = input_list[:19]
        canny_parameters = input_list[19:23]
        resizing_factor = input_list[-1]


    return filtering_parameters, canny_parameters, resizing_factor

def do_u_want_preview():
    print('''\nDo you want to see the image results or let it run in the background? Press the corresponding key
        1- Preview
        2- Background''')
    input2 = int(input())
    return input2


if __name__ == "__main__":
    cap, back = example_or_camera()
    filtering_parameters, canny_parameters, resizing_factor = get_parameters()

    #Code to save the output file
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('output.avi', fourcc, 20.0, (1920,1080))

    back = gp.resize(back, resizing_factor)
    bool_preview = do_u_want_preview()

    if (cap.isOpened() == False):
        print("Error opening Video")

    while True:
        ret, object = cap.read()
        if ret == True:
            object = gp.resize(object, resizing_factor)
            mask, res = gp.filtering(back,object,filtering_parameters)
            canny = gp.canny(res, canny_parameters)
            cont2, center_list = gp.draw_outer_contours(object,canny, canny_parameters[-1])

            if bool_preview == 1:
                contour = gp.draw_contour(object, canny)
                imgs = [back, object, res, mask, canny, contour, cont2]
                titles = ['Background', 'Packs', "Subtraction", 'Mask', 'Canny', 'Contours', 'Boxed Contours']

                canvas = gp.show_images(imgs, titles)
                # out.write(cont2)
                cv.namedWindow("video", cv.WINDOW_NORMAL)
                cv.imshow("video", canvas)
                cv.resizeWindow('video', 1800, 1000)

            gp_out.send_results(center_list)

            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break

    cap.release()
    # out.release()
    cv.destroyAllWindows()

