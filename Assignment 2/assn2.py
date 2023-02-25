import numpy as np
import cv2
from features import extract_hog


def load_average_color_with_bias(X_data):
    """get average color of each image. Add bias dimension at the end with value 1.

    Arguments:
        X_data: numpy array of size (N, H, W, 3)

    Outputs:
        output: numpy array of size (N, 4)
    """
    X_data = X_data.copy()
    N = X_data.shape[0]
    output = np.zeros([N, 4], dtype=X_data.dtype)

    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###

    return output


def load_flatten(X_data):
    """flatten the data.

    Arguments:
        X_data: numpy array of size (N, H * W, D)

    Outputs:
        output: numpy array of size (N * H * W, D)
    """
    X_data = X_data.copy()
    N, HW, D = X_data.shape
    X_data = X_data.copy()

    ### START YOUR CODE HERE ###
    output = X_data.reshape((N * HW), D)
    ### END YOUR CODE HERE ###
    return output


def load_histogram_with_bias(X_data, centroids):
    """given centroid, assign label to each of the keypoints. Draw Histogram

    Arguments:
        X_data: numpy array of size (N, P, D), where N is number of images,
                P is number of keypoints, and D is dimension of features
        centroids: numpy of array of size (K, D), where K is number of centroids.

    Outputs:
        X_hist: numpy array of size (N, K+1), where X_hist[i,j] contains number of
                keypoints from image i that is closest to centroid[j].
                X_hist[:, K] should be 1 for bias.
    """
    X_data, centroids = X_data.copy(), centroids.copy()
    N, P, D = X_data.shape
    K, D = centroids.shape
    X_hist = np.zeros([N, K + 1])

    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###

    return X_hist


def load_hog_representation_with_bias(X_data, cell_size, block_size):
    """get hog_representation

    Arguments:
        X_data: numpy array of size (N, H, W, 3), where N is number of images
        cell_size, block_size: Parameter for HoG

    Outputs:
        X_hog: numpy array of size (N, K+1). Bias dimension at the end.
    """

    X_data = X_data.copy()
    N, H, W = X_data.shape[:3]

    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###

    return X_hog


def load_vector_image_with_bias(X_train, X_val, X_test):
    """Reshape the image data into rows
       Normalize the data by subtracting the mean training image from all images.
       Add bias dimension and transform into columns

    Arguments:
        X_data: numpy array of size (N, H, W, 3), where N is number of images
        cell_size, block_size: Parameter for HoG

    Outputs:
        output: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
    """
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    N_train, N_val, N_text = X_train.shape[0], X_val.shape[0], X_test.shape[0]

    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###

    return X_train, X_val, X_test
