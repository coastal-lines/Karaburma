import random
import cv2
from decimal import Decimal
import imutils
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from karaburma.utils.image_processing.filters_helper import try_threshold


def get_contours_by_different_thresholds(grayscale_image, needs_to_remove_similar=False, block_size_min=3, block_size_max=11):
    binary_local, mean_binary, minimum_binary, otsu_binary, li_binary, isodata_binary, triangle_binary, yen_binary = try_threshold(grayscale_image, block_size_min, block_size_max)

    all_contours = []
    for binary_l in binary_local:
        try:
            contours, hierarchy = get_contours_by_canny(binary_l, 0, 255)
            all_contours.extend(contours)
        except:
            continue

    try:
        mean_binary_contours, _ = get_contours_by_canny(mean_binary, 0, 255)
        all_contours.extend(mean_binary_contours)
    except:
        print("error mean_binary_contours")

    try:
        minimum_binary_contours, _ = get_contours_by_canny(minimum_binary, 0, 255)
        all_contours.extend(minimum_binary_contours)
    except:
        print("error minimum_binary_contours")

    try:
        otsu_binary_contours, _ = get_contours_by_canny(otsu_binary, 0, 255)
        all_contours.extend(otsu_binary_contours)
    except:
        print("error otsu_binary_contours")

    try:
        li_binary_contours, _ = get_contours_by_canny(li_binary, 0, 255)
        all_contours.extend(li_binary_contours)
    except:
        print("error li_binary_contours")

    try:
        isodata_binary_contours, _ = get_contours_by_canny(isodata_binary, 0, 255)
        all_contours.extend(isodata_binary_contours)
    except:
        print("error isodata_binary_contours")

    try:
        triangle_binary_contours, _ = get_contours_by_canny(triangle_binary, 0, 255)
        all_contours.extend(triangle_binary_contours)
    except:
        print("error triangle_binary_contours")

    try:
        yen_binary_contours, _ = get_contours_by_canny(yen_binary, 0, 255)
        all_contours.extend(yen_binary_contours)
    except:
        print("error yen_binary_contours")

    rectangles = []
    for i in range(len(all_contours)):
        x, y, w, h = cv2.boundingRect(all_contours[i])
        rectangles.append((x, y, w, h))

    return all_contours

def get_contours(image_bw):
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def get_contours_external_only(image_bw):
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, hierarchy

def get_canny(image_bw, lower_threshold, upper_threshold):
    detected_edges = cv2.Canny(image_bw, lower_threshold, upper_threshold)
    return detected_edges

def get_contours_by_canny(image_bw, lower_threshold, upper_threshold, external_only=False):
    detected_edges = cv2.Canny(image_bw, lower_threshold, upper_threshold)

    #CV_RETR_LIST - without grouping
    #CV_CHAIN_APPROX_SIMPLE —
    if(external_only):
        contours, hierarchy = get_contours_external_only(detected_edges)
    else:
        contours, hierarchy = cv2.findContours(detected_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours, hierarchy

def get_contours_by_canny_after_approximation(image_bw, lower_threshold, upper_threshold, eps, approximation):
    contours, hierarchy = get_contours_by_canny(image_bw, lower_threshold, upper_threshold)

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

def get_contour_length(contour):
    return cv2.arcLength(contour, True)

def get_match_shapes(contour1, contour2):
    value = cv2.matchShapes(contour1, contour2, 1, 0.0)
    return Decimal(value)

def get_box_from_contour(contour):
    rect = cv2.minAreaRect(contour) #
    box = cv2.boxPoints(rect) #
    box = np.int0(box) #

    return box

def convert_contours_to_rectangles(cnt):
    bounding_boxes = []
    for contour in cnt:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, x + w, y + h))

    return bounding_boxes

def draw_rectangle_by_contours(image, contours, color=(0, 255, 0)):
    if(len(contours) > 0):
        for cnt in contours:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect
            point1 = (x, y)
            point2 = (x + w, y + h)
            cv2.rectangle(image, point1, point2, color, 1)
    else:
        print("Contours were not found")

    return image

def draw_rectangle_by_rectangles(image, rectangles, color=(0, 255, 0)):
    for rect in rectangles:
        x1, y1, x2, y2 = rect
        point1 = (x1, y1)
        point2 = (x2, y2)
        cv2.rectangle(image, point1, point2, color, 1)

def draw_rectangle_by_point(image, p1, p2, color=(0, 255, 0), thicknes = 1):
    cv2.rectangle(image, p1, p2, color, thicknes)

def draw_rectangle(image, startX, startY, endX, endY, color=(0, 255, 0)):
    cv.rectangle(image, (startX, startY), (endX, endY), color, 5)

def draw_rectangle_by_xywh(image, x, y, w, h, color=(0, 255, 0), thicknes = 1):
    cv.rectangle(image, (x, y), (x + w, y + h), color, thicknes)

def draw_rectangle_by_list_xywh(image, rectangles, color=(0, 255, 0), thicknes=1):
    for i in range(len(rectangles)):
        x, y, w, h = rectangles[i][0],rectangles[i][1],rectangles[i][2],rectangles[i][3]
        cv.rectangle(image, (x, y), (x + w, y + h), color, thicknes)

    return image

def draw_filled_rectangle_with_frame(image, label, x, y, colour=(0, 0, 0)):
    shift = 0
    text_size = cv2.getTextSize(text=label, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
    cv2.rectangle(image, (x - shift, y + shift), (x + (text_size[0] + shift), y - text_size[1] - shift), (255, 255, 255), 4)
    cv2.rectangle(image, (x - shift, y + shift), (x + (text_size[0] + shift), y - text_size[1] - shift), colour, -1)

def draw_rectangle_and_label(image, label, probability, x, y, w, h, colour=(255, 255, 255), thicknes=1):
    point_1_rectangle = (x, y)
    point_2_rectangle = (x + w, y + h)

    #draw rectangle for element
    cv.rectangle(image, point_1_rectangle, point_2_rectangle, colour, thicknes)

    #shifting for big lements
    if (h > 100):
        y = y + (h // 2) + random.randint(7, 14)

    point_1_line1 = (x + w, y)
    point_2_line1 = (x + w + 20, y - 10)

    #draw line
    cv2.line(image, point_1_line1, point_2_line1, colour, 1)

    #draw label
    draw_filled_rectangle_with_frame(image, label, x + w + 20, y - 10)
    cv2.putText(image, label, (x + w + 20, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)

def draw_rectangle_and_label_for_element(image, element, color=(255, 0, 255), thicknes=1):
    draw_rectangle_and_label(image,
        element.get_label(),
        element.get_prediction_value(),
        element.get_roi_element().get_x(),
        element.get_roi_element().get_y(),
        element.get_roi_element().get_w(),
        element.get_roi_element().get_h(),
        color,
        thicknes)

def filter_very_similar_contours(rectangles, threshold=0.5):
    rectangles = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rectangles])
    filtered_rectangles = imutils.object_detection.non_max_suppression(np.array(rectangles), probs=None, overlapThresh=threshold)
    filtered_converted_rectangles = np.array([[x, y, x2 - x, y2 - y] for (x, y, x2, y2) in filtered_rectangles])

    return filtered_converted_rectangles

def calculate_similarity(contour1, contour2):
    # Calculate the Hu Moments for the two contours
    moments1 = cv2.moments(contour1)
    moments2 = cv2.moments(contour2)
    hu_moments1 = cv2.HuMoments(moments1)
    hu_moments2 = cv2.HuMoments(moments2)

    # Calculate the similarity between the Hu Moments
    similarity = cv2.matchShapes(hu_moments1, hu_moments2, cv2.CONTOURS_MATCH_I2, 0)

    return similarity

def remove_similar_contours(contours, threshold):
    filtered_contours = []

    for contour in contours:
        is_similar = False
        for filtered_contour in filtered_contours:
            similarity = calculate_similarity(contour, filtered_contour)
            print("similarity", similarity)
            if similarity < threshold:
                is_similar = True
                break
        if not is_similar:
            filtered_contours.append(contour)

    return filtered_contours

def get_x_center(x: int, w: int) -> int:
    return x + (w // 2)

def get_y_center(y: int, h: int) -> int:
    return y + (h // 2)

def get_rect_centroid(rect):
    x, y, w, h = rect
    cX = get_x_center(x, w)
    cY = get_y_center(y, h)
    return cX, cY

def extract_xywh(contour, index):
    return contour[index][0], contour[index][1], contour[index][2], contour[index][3]