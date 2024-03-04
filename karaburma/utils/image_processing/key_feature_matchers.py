import cv2


def get_bf_good_matches(descriptors1, descriptors2, n=20):
    # Use a brute-force matcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort the matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 'n' matches (you can adjust 'n' as needed)
    good_matches = matches[:n]

    return good_matches