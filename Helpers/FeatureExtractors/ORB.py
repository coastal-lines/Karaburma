import cv2
from Helpers.ImageConverters import ImageConverters

class ORB():
    def PrepareKeypointsAndDescriptorsByORB(originalImage):
        query_img = cv2.imread('c:\Temp\Photos\Tests.bmp')  # queryImage
        query_img_bw = ImageConverters.ConvertToBW(query_img)

        # Initialize the ORB detector algorithm
        features = 500
        orb = cv2.ORB_create(features)

        # Now detect the keypoints and compute the descriptors for the query image
        queryKeypoints, queryDescriptors = orb.detectAndCompute(query_img_bw, None)

        return queryKeypoints, queryDescriptors

    def PrepareCustomKeypointsAndDescriptorsByORB(originalImage):
        query_img = cv2.imread('c:\Temp\Photos\Tests.bmp')  # queryImage
        query_img_bw = ImageConverters.ConvertToBW(query_img)

        # Initialize the ORB detector algorithm
        features = 500
        orb = cv2.ORB_create(features)

        kp1 = cv2.KeyPoint(54, 758, 1)
        kp2 = cv2.KeyPoint(164, 546, 1)
        kp3 = cv2.KeyPoint(162, 545, 1)
        kp4 = cv2.KeyPoint(160, 543, 1)
        userKP = [kp1, kp2, kp3, kp4]

        # Now detect the keypoints and compute the descriptors for the query image
        queryKeypoints, queryDescriptors = orb.compute(query_img_bw, userKP)

        return queryKeypoints, queryDescriptors

    def PrepareMatchesByBFMatcher(queryDescriptors, trainDescriptors):
        # Full bust variables - BFMatcher
        # cv2.NORM_HAMMING - metric for comparing distance between objects
        # crosscheck - object A will be compared with object B only if B is the nearest neighbor of A (this is longer but more accurately)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(queryDescriptors, trainDescriptors)
        # do sort by distance
        matches = sorted(matches, key=lambda x: x.distance)
        print(len(matches))

    def ShowORBMatchesBetweenTwoImages(query_img, queryKeypoints, train_img, trainKeypoints, matches):
        final_img = cv2.drawMatches(query_img, queryKeypoints, train_img, trainKeypoints, matches[:50], query_img)
        final_img = cv2.resize(final_img, (2000, 1000))
        cv2.imshow("Matches", final_img)
        cv2.waitKey(0)