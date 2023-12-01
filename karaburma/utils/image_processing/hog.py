import cv2

def compute_hog(image):
    win_size = (8, 8)  # Choose an appropriate window size
    block_size = (2, 2)  # Choose an appropriate block size
    block_stride = (2, 2)
    cell_size = (2, 2)
    num_bins = 9

    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    hog_features = hog.compute(image)

    return hog_features

def compare_hog_features(hog_features1, hog_features2):
    # Calculate the cosine similarity between the two HOG feature vectors
    similarity = cv2.compareHist(hog_features1, hog_features2, cv2.HISTCMP_CORREL)

    return similarity