import random

import cv2
from decimal import Decimal

import imutils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from lena.utils.image_processing.filters_helper import try_threshold


def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()

def get_contours_by_different_thresholds(grayscale_image, needs_to_remove_similar=False, block_size_min=3, block_size_max=11):
    binary_local, mean_binary, minimum_binary, otsu_binary, li_binary, isodata_binary, triangle_binary, yen_binary = try_threshold(grayscale_image, block_size_min, block_size_max)

    all_contours = []
    for binary_l in binary_local:
        try:
            contours, hierarchy = GetContoursByCanny(binary_l, 0, 255)
            all_contours.extend(contours)
        except:
            continue

    try:
        mean_binary_contours, _ = GetContoursByCanny(mean_binary, 0, 255)
        all_contours.extend(mean_binary_contours)
    except:
        print("error mean_binary_contours")

    try:
        minimum_binary_contours, _ = GetContoursByCanny(minimum_binary, 0, 255)
        all_contours.extend(minimum_binary_contours)
    except:
        print("error minimum_binary_contours")

    try:
        otsu_binary_contours, _ = GetContoursByCanny(otsu_binary, 0, 255)
        all_contours.extend(otsu_binary_contours)
    except:
        print("error otsu_binary_contours")

    try:
        li_binary_contours, _ = GetContoursByCanny(li_binary, 0, 255)
        all_contours.extend(li_binary_contours)
    except:
        print("error li_binary_contours")

    try:
        isodata_binary_contours, _ = GetContoursByCanny(isodata_binary, 0, 255)
        all_contours.extend(isodata_binary_contours)
    except:
        print("error isodata_binary_contours")

    try:
        triangle_binary_contours, _ = GetContoursByCanny(triangle_binary, 0, 255)
        all_contours.extend(triangle_binary_contours)
    except:
        print("error triangle_binary_contours")

    try:
        yen_binary_contours, _ = GetContoursByCanny(yen_binary, 0, 255)
        all_contours.extend(yen_binary_contours)
    except:
        print("error yen_binary_contours")


    #rectangles = np.zeros((len(all_contours), 4), dtype=np.int32)
    rectangles = []
    for i in range(len(all_contours)):
        x, y, w, h = cv2.boundingRect(all_contours[i])
        #rectangles[i] = (x, y, x + w, y + h)
        rectangles.append((x, y, w, h))
        #rectangles.append((x, y, x + w, y + h))

    #if(needs_to_remove_similar):
    #    rectangles = non_max_suppression(rectangles, probs=None, overlapThresh=0.9)

    #return all_contours, rectangles
    return all_contours

def get_contours(image_bw):
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def get_contours_external_only(image_bw):
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def GetCanny(image_bw, lower_threshold, upper_threshold):
    detected_edges = cv2.Canny(image_bw, lower_threshold, upper_threshold)
    return detected_edges

def GetContoursByCanny(image_bw, lower_threshold, upper_threshold, external_only=False):
    detected_edges = cv2.Canny(image_bw, lower_threshold, upper_threshold)
    #CV_RETR_LIST - without grouping
    #CV_CHAIN_APPROX_SIMPLE —
    if(external_only):
        contours, hierarchy = get_contours_external_only(detected_edges)
    else:
        contours, hierarchy = cv2.findContours(detected_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def GetContoursByCannyAfterApproximation(image_bw, lower_threshold, upper_threshold, eps, approximation):

    contours, hierarchy = GetContoursByCanny(image_bw, lower_threshold, upper_threshold)

    new_contours = []
    for cnt in contours:
        arclen = cv2.arcLength(cnt, True)
        epsilon = arclen * eps
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        if len(approx) < approximation:
            new_contours.append(cnt)

    return new_contours

def get_contours_after_approximation(image_bw: np.array, approx: int, epsilon: float=0.05):
    all_contours, hierarchy = get_contours(image_bw)

    filtered_rectangles = []

    for j in range(len(all_contours)):
        peri = cv2.arcLength(all_contours[j], True)
        current_approx = cv2.approxPolyDP(all_contours[j], epsilon * peri, True)
        x, y, w, h = cv2.boundingRect(all_contours[j])
        if len(current_approx) == approx:
            filtered_rectangles.append((x, y, w, h))

    return filtered_rectangles

def GetContourLength(contour):
    return cv2.arcLength(contour, True)

def GetMatchShapes(contour1, contour2):
    value = cv2.matchShapes(contour1, contour2, 1, 0.0)
    return Decimal(value)

def GetBoxFromContour(contour):
    rect = cv2.minAreaRect(contour) #
    box = cv2.boxPoints(rect) #
    box = np.int0(box) #
    return box

def ConvertContoursToRectangles(cnt):
    bounding_boxes = []
    for contour in cnt:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))
        #bounding_boxes.append(x, y, x + w, y + h)

    return bounding_boxes

def DrawRectangleByContours(image, contours, color=(0, 255, 0)):
    if(len(contours) > 0):
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            #print(x, y, w, h)
            point1 = (x, y)
            point2 = (x + w, y + h)
            cv2.rectangle(image, point1, point2, color, 1)
    else:
        print("Contours were not found")

    return image

def DrawRectangleByRectangles(image, rectangles, color=(0, 255, 0)):
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        point1 = (x1, y1)
        point2 = (x2, y2)
        cv2.rectangle(image, point1, point2, color, 1)

def DrawRectangleByPoint(image, p1, p2, color=(0, 255, 0), thicknes = 1):
    cv2.rectangle(image, p1, p2, color, thicknes)

def DrawRectangle(image, startX, startY, endX, endY, color=(0, 255, 0)):
    cv.rectangle(image, (startX, startY), (endX, endY), color, 5)

def DrawRectangleByXYWH(image, x, y, w, h, color=(0, 255, 0), thicknes = 1):
    cv.rectangle(image, (x, y), (x + w, y + h), color, thicknes)

def DrawRectangleByListXYWH(image, rectangles, color=(0, 255, 0), thicknes=1):
    for i in range(len(rectangles)):
        x, y, w, h = rectangles[i][0],rectangles[i][1],rectangles[i][2],rectangles[i][3]
        cv.rectangle(image, (x, y), (x + w, y + h), color, thicknes)

    return image

def draw_filled_rectangle_with_frame(image, label, x, y, colour=(0, 0, 0)):
    shift = 0
    text_size = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
    cv2.rectangle(image, (x - shift, y + shift), (x + (text_size[0] + shift), y - text_size[1] - shift), (255, 255, 255), 4)
    cv2.rectangle(image, (x - shift, y + shift), (x + (text_size[0] + shift), y - text_size[1] - shift), colour, -1)