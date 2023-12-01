from abc import ABC, abstractmethod
import cv2
import numpy as np

from karaburma.utils import general_helpers
from karaburma.utils.image_processing import ocr_helper, filters_helper, orb_descriptor, key_feature_matchers, homography

#TODO -> tuple[int, int]

class DisplacementBase(ABC):
    @abstractmethod
    def calculate_displacement(self, before, after=None):
        pass

class OcrVerticalDisplacement(DisplacementBase):
    def calculate_displacement(self, before, after=None):
        return ocr_helper.calculate_scrolling_shift_by_text_position(before)

class OrbBfHomographicalDisplacement(DisplacementBase):
    def calculate_displacement(self, before, after=None):

        # Detect ORB keypoints and descriptors in both images
        keypoints1, descriptors1 = orb_descriptor.create_orb(before)
        keypoints2, descriptors2 = orb_descriptor.create_orb(after)

        good_matches = key_feature_matchers.get_bf_good_matches(descriptors1, descriptors2)

        # Extract the matched keypoints
        points1 = np.float32([keypoints1[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32([keypoints2[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)

        # Use the findHomography function to find the perspective transformation
        # This assumes a perspective transformation (shift and rotation)
        M, _ = homography.calculate_images_transformation(points1, points2)

        # Extract the translation values
        x_displacement = M[0, 2]
        y_displacement = M[1, 2]

        print(f"Horizontal displacement: {x_displacement} pixels")
        print(f"Vertical displacement: {y_displacement} pixels")

        average_offset_x = abs(general_helpers.custom_round(x_displacement))
        average_offset_y = abs(general_helpers.custom_round(y_displacement))

        return average_offset_x, average_offset_y

class AkazeKeyFeaturesDisplacement(DisplacementBase):
    def calculate_displacement(self, before, after=None):
        # Akaze descriptor and detector
        akaze = cv2.AKAZE_create()

        # Find key points and their descriptord
        kp1, des1 = akaze.detectAndCompute(before, None)
        kp2, des2 = akaze.detectAndCompute(after, None)

        # Using BFMatcher for matching descriptors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply metric 'ratio test'
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        # Using matched points for calculate shifting
        sum_x = 0
        sum_y = 0
        for match in good_matches:
            sum_x += kp2[match.trainIdx].pt[0] - kp1[match.queryIdx].pt[0]
            sum_y += kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1]

        # Average of shifting
        if(len(good_matches) > 0):
            average_offset_x = abs(general_helpers.custom_round(sum_x / len(good_matches)))
            average_offset_y = abs(general_helpers.custom_round(sum_y / len(good_matches)))
        else:
            average_offset_x = 0
            average_offset_y = 0

        print("Shift across X:", average_offset_x)
        print("Shift across Y:", average_offset_y)

        return average_offset_x, average_offset_y

class OpticalFlowDisplacement(DisplacementBase):
    def calculate_displacement(self, before, after=None):

        before = filters_helper.convert_to_grayscale(before)
        after = filters_helper.convert_to_grayscale(after)

        score = general_helpers.calculate_similarity(before, after)

        if (score < 1.0):
            #feature_params = dict(maxCorners=300, qualityLevel=0.01, minDistance=3, blockSize=99)

            # good for tables
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=28, blockSize=7)
            p0 = cv2.goodFeaturesToTrack(before, mask=None, **feature_params)

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(before, after, p0, None)

            # Find displacement
            displacement = np.mean(p1 - p0, axis=0)
            x_displacement = abs(general_helpers.custom_round(displacement[0][0]))
            y_displacement = abs(general_helpers.custom_round(displacement[0][1]))

            return x_displacement, y_displacement

        else:
            print("Exception")

class DisplacementManager:
    def __init__(self, displacement_strategy: DisplacementBase):
        self.displacement_strategy = displacement_strategy

    def calculate_displacement(self, before, after=None):
        if after is None and self.displacement_strategy is not OcrVerticalDisplacement:
            #TODO
            print("error. empty after")

        return self.displacement_strategy.calculate_displacement(before, after)