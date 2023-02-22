import cv2
import numpy as np


def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # the default border handling provided by cv2.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.

    half_width = int(3 * sigma)
    width = (2 * half_width) + 1

    ## Derivative of the guassian as calculated in 2a

    norm = 1 / (np.sqrt(2 * np.pi) * (sigma**3))
    filter_x = np.zeros((1, width))
    filter_y = np.zeros((width, 1))

    # filter2d coorelates, not convolve, need to flip the matrix first and then coorelate for convolution
    for x in range(-half_width, half_width + 1):
        filter_x[0, half_width - x] = (
            (-x) * norm * (np.e ** ((-((x) ** 2)) / (2 * (sigma**2))))
        )

    for y in range(-half_width, half_width + 1):
        filter_y[half_width - y, 0] = (
            (-y) * norm * (np.e ** ((-((y) ** 2)) / (2 * (sigma**2))))
        )

    Fx = cv2.filter2D(src=im, ddepth=-1, kernel=filter_x)
    Fy = cv2.filter2D(src=im, ddepth=-1, kernel=filter_y)
    return Fx, Fy


def edgeStrengthAndOrientation(Fx, Fy):
    # Given horizontal and vertical gradients for an image, computes the edge
    # strength and orientation images.
    #
    # Fx: 2D double array with shape (height, width). The horizontal gradients.
    # Fy: 2D double array with shape (height, width). The vertical gradients.
    # Returns:
    # F: 2D double array with shape (height, width). The edge strength
    #        image.
    # D: 2D double array with shape (height, width). The edge orientation
    # image.

    F = np.sqrt(np.square(Fx) + np.square(Fy))
    D = np.arctan(np.divide(Fy, Fx + 1e-13)) % np.pi  # add epsilon to account for
    # runtime warning when Fx is 0

    return F, D


def suppression(F, D):
    # Runs nonmaximum suppression to create a thinned edge image.
    #
    # F: 2D double array with shape (height, width). The edge strength values
    #    for the input image.
    # D: 2D double array with shape (height, width). The edge orientation
    #    values for the input image.
    # Returns:
    # I: 2D double array with shape (height, width). The output thinned
    #        edge image.
    pi = np.pi
    directions = [0, pi / 4, pi / 2, (3 * pi) / 4, pi]
    height, width = np.shape(F)
    I = np.zeros((height, width))

    # find the strongest edge among neighbors per pixel
    for i in range(height):
        for j in range(width):
            val = D[i][j]
            DStar = directions[np.argmin(np.abs(directions - val))]

            # assume I[i][j] = F[i][j] until proven otherwise
            I[i][j] = F[i][j]

            # D* = 0 or pi
            if (DStar == 0) or (DStar == pi):
                if j == (width - 1):
                    if F[i][j] < F[i][j - 1]:
                        I[i][j] = 0
                elif j == 0:
                    if F[i][j] < F[i][j + 1]:
                        I[i][j] = 0
                else:
                    if F[i][j] < F[i][j + 1] or F[i][j] < F[i][j - 1]:
                        I[i][j] = 0

            # D* = pi/4
            elif DStar == (pi) / 4:
                if (i == (height - 1) or j == (width - 1)) and i != 0:
                    if F[i][j] < F[i - 1][j - 1]:
                        I[i][j] = 0
                elif (i == 0 or j == 0) and j != (width - 1):
                    if F[i][j] < F[i + 1][j + 1]:
                        I[i][j] = 0
                elif (i != (height - 1) and j != 0) and (i != 0 and j != width - 1):
                    if F[i][j] < F[i + 1][j + 1] or F[i][j] < F[i - 1][j - 1]:
                        I[i][j] = 0

            # for D* = pi/2
            elif DStar == pi / 2:
                if i == (height - 1):
                    if F[i][j] < F[i - 1][j]:
                        I[i][j] = 0
                elif i == 0:
                    if F[i][j] < F[i + 1][j]:
                        I[i][j] = 0
                else:
                    if F[i][j] < F[i + 1][j] or F[i][j] < F[i - 1][j]:
                        I[i][j] = 0

            # for D* = 3pi/2
            elif DStar == (3 * pi) / 4:

                if (i == 0 or j == (width - 1)) and j != 0 and i != (height - 1):
                    if F[i][j] < F[i + 1][j - 1]:
                        I[i][j] = 0
                elif (j == 0 or i == (height - 1)) and i != 0 and j != (width - 1):
                    if F[i][j] < F[i - 1][j + 1]:
                        I[i][j] = 0
                elif (i != 0 and j != 0) and (i != (height - 1) and j != (width - 1)):
                    if F[i][j] < F[i + 1][j - 1] or F[i][j] < F[i - 1][j + 1]:
                        I[i][j] = 0

    return I


def hysteresisThresholding(I, D, tL, tH):
    # Runs hysteresis thresholding on the input image.
    # I: 2D double array with shape (height, width). The input's edge image
    #    after thinning with nonmaximum suppression.
    # D: 2D double array with shape (height, width). The edge orientation
    #    image.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.
    # Returns:
    # edgeMap: 2D binary array with shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.

    # normalize I so that max value is 1
    maximum = np.amax(I)
    minimum = np.amin(I)
    height, width = np.shape(I)
    for i in range(height):
        for j in range(width):
            I[i][j] = (I[i][j] - minimum) / (maximum - minimum)

    height, width = np.shape(I)
    visited = np.zeros((height, width))
    edgeMap = np.zeros((height, width))
    pi = np.pi
    directions = [0, pi / 4, pi / 2, (3 * pi) / 4, pi]
    edge = np.argwhere(I > tH)
    x = len(edge) - 1  # keep track of iteration of edge pixels searched

    # mark pixels that are already found from tH as visited
    for x in range(len(edge)):
        i, j = edge[x][0], edge[x][1]
        visited[i][j] = -1

    while x >= 0:
        # a pixel that forms an edge
        i, j = edge[x][0], edge[x][1]

        # decriment x for while loop
        x = x - 1

        val = D[i][j]
        DStar = directions[np.argmin(np.abs(directions - val))]

        # D* = 0 or pi check pi/2
        if (DStar == 0) or (DStar == pi):

            if i == (height - 1):
                if I[i - 1][j] > tL and visited[i - 1][j] != -1:
                    edge = np.append(edge, [[i - 1, j]], axis=0)
                    visited[i - 1][j] = -1
                    x = x + 1
                else:
                    visited[i - 1][j] = -1

            elif i == 0:
                if I[i + 1][j] > tL and visited[i + 1][j] != -1:
                    edge = np.append(edge, [[i + 1, j]], axis=0)
                    visited[i + 1][j] = -1
                    x = x + 1
                else:
                    visited[i + 1][j] = -1

            else:
                if I[i + 1][j] > tL and visited[i + 1][j] != -1:
                    edge = np.append(edge, [[i + 1, j]], axis=0)
                    visited[i + 1][j] = -1
                    x = x + 1

                if I[i - 1][j] > tL and visited[i - 1][j] != -1:
                    edge = np.append(edge, [[i - 1, j]], axis=0)
                    visited[i - 1][j] = -1
                    x = x + 1
                else:
                    visited[i + 1][j] = -1
                    visited[i - 1][j] = -1

        # D* = pi/4 check 3pi/4
        elif DStar == pi / 4:
            if (i == 0 or j == (width - 1)) and j != 0 and i != (height - 1):
                if I[i + 1][j - 1] > tL and visited[i + 1][j - 1] != -1:
                    edge = np.append(edge, [[i + 1, j - 1]], axis=0)
                    visited[i + 1][j - 1] = -1
                    x = x + 1
                else:
                    visited[i + 1][j - 1] = -1
            elif (j == 0 or i == (height - 1)) and i != 0 and j != (width - 1):
                if I[i - 1][j + 1] > tL and visited[i - 1][j + 1] != -1:
                    edge = np.append(edge, [[i - 1, j + 1]], axis=0)
                    visited[i - 1][j + 1] = -1
                    x = x + 1
                else:
                    visited[i - 1][j + 1] = -1
            elif (i != 0 and j != 0) and (i != (height - 1) and j != (width - 1)):
                if I[i + 1][j - 1] > tL and visited[i + 1][j - 1] != -1:
                    edge = np.append(edge, [[i + 1, j - 1]], axis=0)
                    visited[i + 1][j - 1] = -1
                    x = x + 1
                if I[i - 1][j + 1] > tL and visited[i - 1][j + 1] != -1:
                    edge = np.append(edge, [[i - 1, j + 1]], axis=0)
                    visited[i - 1][j + 1] = -1
                    x = x + 1
                else:
                    visited[i - 1][j + 1] = -1
                    visited[i + 1][j - 1] = -1

        # D* = pi/2 check 0
        elif DStar == pi / 2:
            if j == (width - 1):
                if I[i][j - 1] > tL and visited[i][j - 1] != -1:
                    edge = np.append(edge, [[i, j - 1]], axis=0)
                    visited[i][j - 1] = -1
                    x = x + 1
                else:
                    visited[i][j - 1] = -1

            elif j == 0:
                if I[i][j + 1] > tL and visited[i, j + 1] != -1:
                    edge = np.append(edge, [[i, j + 1]], axis=0)
                    visited[i][j + 1] = -1
                    x = x + 1
                else:
                    visited[i][j + 1] = -1
            else:
                if I[i][j + 1] > tL and visited[i][j + 1] != -1:
                    edge = np.append(edge, [[i, j + 1]], axis=0)
                    visited[i][j + 1] = -1
                    x = x + 1

                if I[i][j - 1] > tL and visited[i][j - 1] != -1:
                    edge = np.append(edge, [[i, j - 1]], axis=0)
                    visited[i][j - 1] = -1
                    x = x + 1

                else:
                    visited[i][j - 1] = -1
                    visited[i][j + 1] = -1

        # D* = 3pi/4 check pi/4
        elif DStar == 3 * pi / 4:
            if (i == (height - 1) or j == (width - 1)) and i != 0:
                if I[i - 1][j - 1] > tL and visited[i - 1][j - 1] != -1:
                    edge = np.append(edge, [[i - 1, j - 1]], axis=0)
                    visited[i - 1][j - 1] = -1
                    x = x + 1
                else:
                    visited[i - 1][j - 1] = -1
            elif (i == 0 or j == 0) and j != (width - 1):
                if I[i + 1][j + 1] > tL and visited[i + 1][j + 1] != -1:
                    edge = np.append(edge, [[i + 1, j + 1]], axis=0)
                    visited[i + 1][j + 1] = -1
                    x = x + 1
                else:
                    visited[i + 1][j + 1] = -1
            elif (i != (height - 1) and j != 0) and (i != 0 and j != width - 1):
                if I[i + 1][j + 1] > tL and visited[i + 1][j + 1] != -1:
                    edge = np.append(edge, [[i + 1, j + 1]], axis=0)
                    visited[i + 1][j + 1] = -1
                    x = x + 1
                if I[i - 1][j - 1] > tL and visited[i - 1][j - 1] != -1:
                    edge = np.append(edge, [[i - 1, j - 1]], axis=0)
                    visited[i - 1][j - 1] = -1
                    x = x + 1
                else:
                    visited[i - 1][j - 1] = -1
                    visited[i + 1][j + 1] = -1

    for x in range(len(edge)):
        i, j = edge[x][0], edge[x][1]
        edgeMap[i][j] = 1

    return edgeMap


def cannyEdgeDetection(im, sigma, tL, tH):
    # Runs the canny edge detector on the input image. This function should
    # not duplicate your implementations of the edge detector components. It
    # should just call the provided helper functions, which you fill in.
    #
    # IMPORTANT: We have broken up the code this way so that you can get
    # better partial credit if there is a bug in the implementation. Make sure
    # that all of the work the algorithm does is in the proper helper
    # functions, and do not change any of the provided interfaces. You
    # shouldn't need to create any new .py files, unless they are for testing
    # these provided functions.
    #
    # im: 2D double array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.
    # Returns:
    # edgeMap: 2D binary image of shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.
    # TODO: Implement me!

    # read image in and make it grayscale and
    # floating point
    img = cv2.imread(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    Fx, Fy = filteredGradient(img, sigma)
    F, D = edgeStrengthAndOrientation(Fx, Fy)
    I = suppression(F, D)
    edgeMap = hysteresisThresholding(I, D, tL, tH)

    return edgeMap
