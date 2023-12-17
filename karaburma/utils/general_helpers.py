import copy
import math
import random
import string
import time

import cv2
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pyautogui
import skimage
from skimage.transform import resize
from skimage.color import rgb2gray
#from sklearn.metrics import plot_roc_curve
from skimage.metrics import structural_similarity as ssim
from PIL import Image as PIL_Image
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte

from karaburma.utils import files_helper
from karaburma.utils.image_processing import filters_helper, contours_helper, morphological_helpers
from karaburma.utils.image_processing.filters_helper import try_threshold


def save_file(img, i):
    path = str(i) + ".bmp"
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite(path, skimage.img_as_ubyte(img))

def save_file2(img, path):
    path = path + ".bmp"
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imwrite(path, img)

def save_detected_regions(screenshot, svm_startX, svm_startY, svm_endX, svm_endY, image_number):

    path = str(image_number) + ".bmp"
    width = svm_endX - svm_startX
    height = svm_endY - svm_startY
    roi = screenshot[svm_startY:svm_startY + height, svm_startX:svm_startX + width]
    roi = cv.cvtColor(roi, cv.COLOR_BGR2RGB)
    cv.imwrite(path, skimage.img_as_ubyte(roi))

def show(img, title=""):
    plt.title(title)
    #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.imshow(img, cmap='gray')
    plt.show()

def draw_contours_and_show(img, rectangles):
    show(contours_helper.DrawRectangleByListXYWH(img, rectangles))


def extend_screenshot_by_rigth_border(screenshot, additional_width=50, additional_height=0):
    w = screenshot.shape[1] + additional_width
    h = screenshot.shape[0] + additional_height

    extended_img = np.zeros((h, w, 3), dtype=np.uint8)
    extended_img[:, 0:w - additional_width] = screenshot

    return extended_img

def draw_label(screenshot, x1, y1, label):

    p2 = (x1 + 10, y1 - 10)
    cv2.line(screenshot, (x1, y1), p2, (246, 41, 144), 1)
    cv2.line(screenshot, p2, (p2[0] + 30, p2[1]), (246, 41, 144), 1)
    cv2.putText(screenshot, label, p2, cv2.FONT_HERSHEY_PLAIN, 1, (246, 141, 144), 1, cv2.LINE_AA)

def find_objects_by_contours_and_svm(screenshot, model, rect, categories, dimension, type = "custom", extend=True):

    img_temp = screenshot.copy()

    count = 0
    #get coordinates from each contour -> get roi from original image by these coordinates -> make prediction by trained model
    for i in range(len(rect)):
        x1, y1, x2, y2 = rect[i]

        if(extend):
            shift = 1
            x1 = x1 - shift
            x2 = x2 + (shift*2)
            y1 = y1 - shift
            y2 = y2 + (shift*2)

        roi = img_temp[y1:y2, x1:x2]
        roi = resize(roi, dimension, anti_aliasing=True, mode='reflect')
        roi = rgb2gray(roi)
        roi_flatten = [roi.flatten()]
        pred = categories[model.predict(roi_flatten)[0]]
        #print(pred)
        if (pred == type):
            SaveDetectedRegions(img_temp, x1, y1, x2, y2, i)
            contours_helper.DrawRectangleByPoint(screenshot, (x1, y1), (x2, y2), (246, 41, 144))
            DrawLabel(screenshot, x1, y1, type)
            count += 1

    print("Were found: " + str(count))
    #show(screenshot)

def find_best_contours_for_svm_results(screenshot, svm_contours, dimension, type = "custom"):

    height = dimension[0]
    width = dimension[1]

    #find contours
    img = screenshot.copy()
    img_bw = filters_helper.convert_to_grayscale(img)
    img_bw = filters_helper.GammaCorrection(img_bw, 0.1)
    img_bw = filters_helper.Erosion(img_bw)
    img_bw = filters_helper.blur(img_bw, (3, 3))
    ret, th = filters_helper.threshold(img_bw, 135, 255)
    cnt = contours_helper.GetContoursByCannyAfterApproximation(th, 0, 255, 0.005, 18)
    #draw all contours
    #contours.DrawRectangleByContours(screenshot, cnt, (56, 99, 255))

    #get svm results
    temp_results = []
    image_number = 0
    for svm_c in svm_contours:
        svm_startX, svm_startY, svm_endX, svm_endY = svm_c
        svm_center_x = ((svm_endX - svm_startX) // 2) + svm_startX
        svm_center_y = ((svm_endY - svm_startY) // 2) + svm_startY
        #draw svm contours
        #contours.DrawRectangle(screenshot, svm_startX, svm_startY, svm_endX, svm_endY, (255, 128, 128))
        #SaveDetectedRegions(screenshot, svm_startX, svm_startY, svm_endX, svm_endY, image_number)
        #image_number += 1

        temp_contour = []
        #find by center
        for i in range(len(cnt)):
            x, y, w, h = cv2.boundingRect(cnt[i])
            if((svm_center_x > x and svm_center_x < (x + w)) and (svm_center_y > y and svm_center_y < (y + h))):

                if (type == "checkboxes"):
                    temp_results.append((x, y, w, h))
                    #print(type)
                '''
                if(type == "buttons"):
                    if(w < (width * 5) and h < (height * 2)):
                        temp_results.append((x, y, w, h))

                if (type == "input"):
                    if (w < (width * 10) and h < (height * 3)):
                        temp_results.append((x, y, w, h))

                if (type == "dropdown"):
                    temp_results.append((x, y, w, h))

                if (type == "radiobuttons"):
                    temp_results.append((x, y, w, h))
                    
                #temp_results.append((x, y, w, h))
                '''
    #remove duplicates
    results = list(set([i for i in temp_results]))

    '''
    for r in results:
        x, y, w, h = r
        svm_endX = x + w
        svm_endY = y + h
        #SaveDetectedRegions(screenshot, x, y, svm_endX, svm_endY, image_number)
        image_number += 1
    '''

    print("Detected number is: " + str(len(results)))
    #draw founded results
    contours_helper.DrawRectangleByContours(screenshot, results, (0, 255, 0))

    return results

def draw_label(screenshot, text, point1, point2):

    # draw lines and label
    #cv2.line(screenshot, (svm_endX, svm_startY), (svm_endX + 20, svm_startY + 20), (0, 0, 0), 1)
    #cv2.putText(screenshot, t, (svm_endX + 20, svm_startY + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.line(screenshot, text, point1, point2, (0, 0, 0), 1)
    cv2.putText(screenshot, text, point1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

def get_roi(img, x1,y1,x2,y2):
    roi = img[y1:y2, x1:x2]
    return roi

def fill_image_except_roi(img, start_x1, start_y1, end_x2, end_y2, colour=(0)):
    temp_image = img.copy()
    temp_image[start_y1:end_y2, start_x1:end_x2] = colour
    return temp_image


def show_all_thresholds(image):
    binary_local, mean_binary, minimum_binary, otsu_binary, li_binary, isodata_binary, triangle_binary, yen_binary = try_threshold(image)

    f, axarr = plt.subplots(3, 3)
    axarr[0, 0].imshow(isodata_binary, cmap='gray')
    axarr[0, 0].set_title("Isodata")

    axarr[0, 1].imshow(minimum_binary, cmap='gray')
    axarr[0, 1].set_title("Minimum")

    axarr[0, 2].imshow(binary_local[0], cmap='gray')
    axarr[0, 2].set_title("Binary0")

    axarr[1, 0].imshow(mean_binary, cmap='gray')
    axarr[1, 0].set_title("Mean")

    axarr[1, 1].imshow(otsu_binary, cmap='gray')
    axarr[1, 1].set_title("Otsu")

    axarr[1, 2].imshow(binary_local[1], cmap='gray')
    axarr[1, 2].set_title("Binary1")

    axarr[2, 0].imshow(triangle_binary, cmap='gray')
    axarr[2, 0].set_title("Triangle")

    axarr[2, 1].imshow(yen_binary, cmap='gray')
    axarr[2, 1].set_title("Yen")

    axarr[2, 2].imshow(li_binary, cmap='gray')
    axarr[2, 2].set_title("li_binary")

    plt.show()

def generate_random_string(length):
    letters = string.ascii_letters
    random_string = ''.join(random.choice(letters) for i in range(length))
    return random_string

def calculate_absolute_coordinates(parent_element, x, y):
    return parent_element.get_x() + x, parent_element.get_y() + y

def custom_round(number):
    # Separate the integer and decimal parts
    integer_part = int(abs(number))
    decimal_part = abs(number) - integer_part

    # Custom rounding logic
    if decimal_part < 0.5 or integer_part == 0:
        return integer_part
    else:
        return integer_part + 1

def calculate_similarity(before, after):
    before = filters_helper.convert_to_grayscale(before)
    after = filters_helper.convert_to_grayscale(after)
    ssim_score = ssim(before, after)
    # print(f"SSIM: {ssim_score:.2f}")
    return ssim_score

def do_screenshot():
    return np.array(pyautogui.screenshot())