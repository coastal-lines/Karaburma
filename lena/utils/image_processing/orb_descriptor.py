import cv2
import matplotlib.pyplot as plt
import numpy as np
from lena.utils.image_processing import filters_helper

def create_orb(image, patchSize = 8):
    orb = cv2.ORB_create()
    return orb.detectAndCompute(image, None)

def prepare_keypoints_and_descriptors_by_ORB(query_img):
    #query_img_bw = rgb2gray(query_img)
    query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

    # Initialize the ORB detector algorithm
    orb = cv2.ORB_create(patchSize = 8)

    # Now detect the keypoints and compute the descriptors for the query image
    queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)

    return queryKeypoints, queryDescriptors

def prepare_matches_by_BFMatcher(queryDescriptors, trainDescriptors):
    # Full bust variables - BFMatcher
    # cv2.NORM_HAMMING - metric for comparing distance between objects
    # crosscheck - object A will be compared with object B only if B is the nearest neighbor of A (this is longer but more accurately)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    # do sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    print(len(matches))
    return matches

def show_ORB_matches_between_two_images(query_img, queryKeypoints, train_img, trainKeypoints, matches):
    final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:10], query_img)
    final_img = cv2.resize(final_img, (2000, 1000))
    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)


def match_keypoints(needle_img, original_image, patch_size=2):
    min_match_count = 2

    orb = cv2.ORB_create(edgeThreshold=0, patchSize=patch_size)
    keypoints_needle, descriptors_needle = orb.detectAndCompute(needle_img, None)
    orb2 = cv2.ORB_create(edgeThreshold=0, patchSize=patch_size, nfeatures=2000)
    keypoints_haystack, descriptors_haystack = orb2.detectAndCompute(original_image, None)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)

    search_params = dict(checks=50)

    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_needle, descriptors_haystack, k=2)
    except cv2.error:
        return None, None, [], [], None

    # store all the good matches as per Lowe's ratio test.
    good = []
    points = []

    for pair in matches:
        if len(pair) == 2:
            if pair[0].distance < 0.3 * pair[1].distance:
                good.append(pair[0])

    if len(good) > min_match_count:
        print('match %03d, kp %03d' % (len(good), len(keypoints_needle)))
        for match in good:
            points.append(keypoints_haystack[match.trainIdx].pt)
        # print(points)

    return keypoints_needle, keypoints_haystack, good, points