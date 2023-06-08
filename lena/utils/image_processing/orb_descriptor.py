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