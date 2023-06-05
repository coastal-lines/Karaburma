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