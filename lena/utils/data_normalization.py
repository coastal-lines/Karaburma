from sklearn.preprocessing import MinMaxScaler


def scaling_by_min_max(image):
    return MinMaxScaler().fit_transform(image)