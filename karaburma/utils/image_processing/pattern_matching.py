import math
import cv2
import numpy as np

from karaburma.utils.image_processing import augmentation, filters_helper


def match_template(source: np.array, pattern: np.array, method: int=cv2.TM_CCOEFF_NORMED):
    return cv2.matchTemplate(filters_helper.convert_to_grayscale(source),
                             filters_helper.convert_to_grayscale(pattern),
                             method)

def calculate_min_max(pattern: np.array, source: np.array):
    return cv2.minMaxLoc(match_template(source, pattern))

def multi_match_for_list_patterns(screen, patterns_list, threshold):
    temp_results = []

    for pattern in patterns_list:
        # If RGB
        if (len(pattern.shape) == 3):
            _, w, h = pattern.shape[::-1]
        # If grayscale
        else:
            w, h = pattern.shape[::-1]
        res = match_template(screen, pattern)
        locations = np.swapaxes(np.where(res >= threshold), 0, 1)

        for location in locations:
            print(location[1], location[0], w, h)
            temp_results.append((location[1], location[0], w, h))

    return temp_results

def multi_match_for_list_patterns_with_augmentation(screen, patterns_list, threshold):
    extended_patterns_list = []

    for template in patterns_list:
        w, h = template.shape[::-1]
        shift_x = math.ceil((w / 100) * 5)
        shift_y = math.ceil((h / 100) * 5)

        min_w = math.ceil((w / 100) * 50)
        min_h = math.ceil((h / 100) * 50)

        for i in range(1, 20):
            extended_patterns_list.append(augmentation.bicubic_resize(template, (w + (i * shift_x), h + (i * shift_y))))
            extended_patterns_list.append(augmentation.bicubic_resize(template, (w + (i * shift_x), h)))
            extended_patterns_list.append(augmentation.bicubic_resize(template, (w, h + (i * shift_y))))

            if (w - (i * shift_x) > min_w and h - (i * shift_y) > min_h):
                extended_patterns_list.append(augmentation.bicubic_resize(template, (w - (i * shift_x), h - (i * shift_y))))
                extended_patterns_list.append(augmentation.bicubic_resize(template, (w - (i * shift_x), h)))
                extended_patterns_list.append(augmentation.bicubic_resize(template, (w, h - (i * shift_y))))

    patterns_list.extend(extended_patterns_list)

    return multi_match_for_list_patterns(screen, patterns_list, threshold)