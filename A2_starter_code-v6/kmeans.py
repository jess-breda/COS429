import numpy as np

def calculate_labels(x, centroids):
    """ calculate labels. for every point in x, assign label which is closest to centroids. 
    
    
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
    raise NotImplementedError
    ### YOUR CODE ENDS HERE ###
    
    return labels

def calculate_centroid(x, labels, K):
    """ Calculate new centroid using labels and x. 
    
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
    raise NotImplementedError
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
    x = x.copy()
    N, D = x.shape

    np.random.seed(seed)
    unique_colors = np.unique(x.reshape(-1, D), axis=0)
    idx = np.random.choice(len(unique_colors), K, replace=False)

    # Randomly choose centroids
    centroids = unique_colors[idx, :]

    # Initialize labels
    labels = np.zeros((x.shape[0], ), dtype=np.uint8)

    ### YOUR CODE STARTS HERE ###
    raise NotImplementedError
    ### YOUR CODE ENDS HERE ###
    
    return labels, centroids
