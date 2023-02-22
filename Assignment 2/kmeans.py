import numpy as np


def calculate_labels(x, centroids):
    """calculate labels. for every point in x, assign label which is closest to centroids.


    Arguments
        x: numpy of size (N, D)
        centroids: numpy array of size (K, D)

    Output
        labels : numpy of size (N,)
    """
    x, centroids = x.copy(), centroids.copy()
    N, D = x.shape
    K, D = centroids.shape

    ### YOUR CODE STARTS HERE ###

    # find the euclidean distance using the expanded formula
    # such that sqrt((a+b)^2) = sqrt(a*a + b*b - 2 a*b.T)
    # to allow for mat multiplication
    xx = (x * x).sum(axis=1).reshape(N, 1) * np.ones(shape=(1, K))
    cc = (centroids * centroids).sum(axis=1) * np.ones(shape=(N, 1))
    ED = np.sqrt(xx + cc - (2 * np.dot(x, centroids.T)))

    # assign each pixel (1,..,N) to label(1,..,K) by
    # finding the minimum distance along axis K
    labels = np.argmin(ED, axis=1)

    ### YOUR CODE ENDS HERE ###

    return labels


def calculate_centroid(x, labels, K):
    """Calculate new centroid using labels and x.

    Arguments
        x: numpy of size (N, D)
        labels: numpy of size (N,)
        K: integer.
    Output
        centroids: numpy array of size (K, D)
    """
    x, labels = x.copy(), labels.copy()
    N, D = x.shape
    N = labels.shape
    centroids = np.zeros((K, D))

    ### YOUR CODE STARTS HERE ###

    # only update mean for assigned labels
    ## this is super slow!!
    for k in range(K):
        centroids[k, :] = np.mean(x[labels == k], axis=0)

    ### YOUR CODE ENDS HERE ###

    return centroids


def kmeans(x, K, niter, seed=123):
    """
    x: array of shape (N, D)
    K: integer
    niter: integer

    labels: array of shape (height*width, )
    centroids: array of shape (K, D)

    Note: Be careful with the size of numpy array!
    """

    np.random.seed(seed)
    unique_colors = np.unique(x.reshape(-1, 3), axis=0)
    idx = np.random.choice(len(unique_colors), K, replace=False)

    # Randomly choose centroids
    centroids = unique_colors[idx, :]

    # Initialize labels
    labels = np.zeros((x.shape[0],), dtype=np.uint8)

    ### YOUR CODE STARTS HERE ###
    for ii in range(niter):

        labels = calculate_labels(x, centroids)

        centroids = calculate_centroid(x, labels, K=K)

    ### YOUR CODE ENDS HERE ###

    return labels, centroids
