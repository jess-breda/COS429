import numpy as np
import cv2
import matplotlib.pyplot as plt


############## SIFT ############################################################

def extract_sift(img, step_size=1):
    """
    Extract SIFT features for a given grayscale image. Instead of detecting 
    keypoints, we will set the keypoints to be uniformly distanced pixels.
    Feel free to use OpenCV functions.
    
    Note: Check sift.compute and cv2.KeyPoint

    Args:
        img: Grayscale image of shape (H, W)
        step_size: Size of the step between keypoints.
        
    Return:
        descriptors: numpy array of shape (int(img.shape[0]/step_size) * int(img.shape[1]/step_size), 128)
                     contains sift feature.
    """
    sift = cv2.SIFT_create() # or cv2.xfeatures2d.SIFT_create()
    descriptors = np.zeros((int(img.shape[0]/step_size) * int(img.shape[1]/step_size), 128))

    
    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###

    return descriptors


def extract_sift_for_dataset(data, step_size=1):
    all_features = []
    for i in range(len(data)):
        img = data[i]
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        descriptors = extract_sift(img, step_size)
        all_features.append(descriptors)
    return np.stack(all_features, 0)




############## SPM #############################################################

def spatial_pyramid_matching_with_bias(L, feature, centroids):
    """
    Rebuild the descriptors according to the level of pyramid.
    

    Arg:
        L: Total number of levels in the pyramid. 
        feature: (H, W, 128) numpy array of extracted SIFT features for an image
        centroids: (K, 128) numpy array of the centroids

    Return:
        hist: SPM representation of the SIFT features. (K*int((4**(L+1) - 1)/3) + 1, ) numpy array
              with bias dimension at the end
    """

    ### We provided rough guidelines ###
    # For each level from 0, 1, ..., L

        # For each block (at level 0, block is sized whole image, at level 1
        #                 block is sized [h/2, w/2])
                # Compute histogram of all features within the block using centroids

                # Calculate the weight of the layer. Check equation 1.3 and figure 1.2
                # of https://slazebni.cs.illinois.edu/publications/pyramid_chapter.pdf

                # Append the weighted histogram
    # Normalize the histogram. (subtract mean, divide by std.)
    
    
    H, W = feature.shape[0], feature.shape[1]
    K = centroids.shape[0]
    hist = np.zeros(K*int((4**(L+1) - 1)/3)+1, )
    
    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###
    
    
    return hist


############## HOG #############################################################

def get_differential_filter():
    """
    Define the filters to be applied to compute the image gradients.
    Use sobel filter.

    Returns:
        filter_x: (3, 3) numpy array
        filter_y: (3, 3) numpy array
    """
    filter_x = np.zeros((3, 3))
    filter_y = np.zeros((3, 3))
    
    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###
    
    return filter_x, filter_y


def filter_image(im, filter):
    """
    Apply the given filter to the grayscale image

    Args:
        img: Grayscale image of shape (H, W)

    Returns:
        im_filtered: Grayscale image of shape (H, W)
    """
    im = im.copy()
    m, n = im.shape
    im_filtered = np.zeros((m, n))

    ### START YOUR CODE HERE ###
    raise NotImplementedError
    ### END YOUR CODE HERE ###
    

    return im_filtered


def get_gradient(im_dx, im_dy):
    assert im_dx.shape == im_dy.shape
    m, n = im_dx.shape
    grad_mag = np.zeros((m, n))
    grad_angle = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            grad_mag[i][j] = np.sqrt(im_dx[i][j] ** 2 + im_dy[i][j] ** 2)
            grad_angle[i][j] = np.arctan2(im_dy[i][j], im_dx[i][j])
    return grad_mag, grad_angle


def get_angle_bin(angle):
    angle_deg = angle * 180 / np.pi
    if 0 <= angle_deg < 15 or 165 <= angle_deg < 180: return 0
    elif 15 <= angle_deg < 45: return 1
    elif 45 <= angle_deg < 75: return 2
    elif 75 <= angle_deg < 105: return 3
    elif 105 <= angle_deg < 135: return 4
    elif 135 <= angle_deg < 165: return 5


def build_histogram(grad_mag, grad_angle, cell_size):
    m, n = grad_mag.shape
    M = int(m / cell_size)
    N = int(n / cell_size)
    ori_histo = np.zeros((M, N, 6))
    for i in range(M):
        for j in range(N):
            for x in range(cell_size):
                for y in range(cell_size):
                    angle = grad_angle[i * cell_size + x][j * cell_size + y]
                    mag = grad_mag[i * cell_size + x][j * cell_size + y]
                    bin = get_angle_bin(angle)
                    ori_histo[i][j][bin] += mag

    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    M, N, bins = ori_histo.shape
    e = 0.001
    ori_histo_normalized = np.zeros((M - block_size + 1, N - block_size + 1, bins * block_size * block_size))
    for i in range(M - block_size + 1):
        for j in range(N - block_size + 1):
            unnormalized = []
            for x in range(block_size):
                for y in range(block_size):
                    for z in range(bins):
                        unnormalized.append(ori_histo[i + x][j + y][z])
            unnormalized = np.asarray(unnormalized)
            den = np.sqrt(np.sum(unnormalized ** 2) + e ** 2)
            normalized = unnormalized / den
            for p in range(bins * block_size * block_size):
                ori_histo_normalized[i][j][p] = normalized[p]

    return ori_histo_normalized


def extract_hog(im, cell_size, block_size, plot=False):

    # Process the given image
    im = im.astype('float') / 255.0

    # Extract HOG
    filter_x, filter_y = get_differential_filter()
    im_dx, im_dy = filter_image(im, filter_x), filter_image(im, filter_y)
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    hog = ori_histo_normalized.reshape((-1))

    if plot:
        # Plot the original image and HOG overlayed on it
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(im, cmap='gray'); plt.axis('off')
        plt.subplot(122)
        mesh_x, mesh_y, mesh_u, mesh_v, num_bins = visualize_hog(im, hog, cell_size, block_size)
        plt.imshow(im, cmap='gray', vmin=0, vmax=1); plt.axis('off')
        for i in range(num_bins):
            plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                       color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
        plt.show()

        # Plot the intermediate components
        plt.figure(figsize=(10, 10))
        plt.subplot(141); plt.axis('off')
        plt.imshow(im_dx, cmap='hot', interpolation='nearest')
        plt.subplot(142); plt.axis('off')
        plt.imshow(im_dy, cmap='hot', interpolation='nearest')
        plt.subplot(143); plt.axis('off')
        plt.imshow(grad_mag, cmap='hot', interpolation='nearest')
        plt.subplot(144); plt.axis('off')
        plt.imshow(grad_angle, cmap='hot', interpolation='nearest')
        plt.show()

    return hog


def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    return mesh_x, mesh_y, mesh_u, mesh_v, num_bins
