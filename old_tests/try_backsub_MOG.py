import cv2 as cv
import numpy as np
font = cv.FONT_HERSHEY_SIMPLEX
back = cv.imread('../images/background.jpg')
object = cv.imread('../images/objects.jpg')
#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
# fgbg = cv.createBackgroundSubtractorMOG2()
# fgbg = cv.bgsegm.BackgroundSubtractorGMG()
# fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
# mask = cv.createBackgroundSubtractorKNN(detectShadows=True).apply(back)
fgbg = cv.bgsegm.createBackgroundSubtractorMOG(history = 1)
mask_MOG = fgbg.apply(back)
mask_MOG = fgbg.apply(object)
text_MOG = 'Obtained w/MOG'
mask_MOG_bgr = cv.cvtColor(mask_MOG, cv.COLOR_GRAY2BGR)
mask_MOG = cv.putText(mask_MOG_bgr, text_MOG, (75,2400), font, 3, (0, 255, 255), 8, cv.LINE_AA)

img_conc = np.concatenate((object,mask_MOG_bgr), axis=1)


fgbg = cv.createBackgroundSubtractorMOG2(history = 1)
mask_MOG2 = fgbg.apply(back)
mask_MOG2 = fgbg.apply(object)
text_MOG2 = 'Obtained w/MOG2'
mask_MOG2_bgr = cv.cvtColor(mask_MOG2, cv.COLOR_GRAY2BGR)
mask_MOG2 = cv.putText(mask_MOG2_bgr, text_MOG2, (75,2400), font, 3, (0, 255, 255), 8, cv.LINE_AA)

img_conc = np.concatenate((img_conc,mask_MOG2_bgr), axis=1)


cv.namedWindow("Mask", cv.WINDOW_NORMAL)
cv.imshow('Mask', img_conc)
cv.resizeWindow('Mask', 1200, 600)

k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()