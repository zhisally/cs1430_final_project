import cv2
import numpy as np
# from imutils import contours
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

'''
4/10/21: Function removes background of image input
Takes image (array form) and outputs an image array of same dimensions
but with background removed
'''
def remove_background(image):
    # original = image.copy()

    # Canny Edge Detector (check if need adjust depending on image)
    canny_low = 50 # Started at 15
    canny_high = 150
    mask_color = (0.0,0.0,0.0)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c_edges = cv2.Canny(gray_img, canny_low, canny_high)

    # Removes noise in background
    kernel = np.ones((20, 20), dtype=np.uint8) # Check effects of kernel size on output
    c_edges = cv2.morphologyEx(c_edges, cv2.MORPH_CLOSE, kernel)

    # cv2_imshow(c_edges)

    # c_edges = cv2.dilate(c_edges, None)
    # c_edges = cv2.erode(c_edges, None)

    mask = np.zeros(c_edges.shape, dtype=np.uint8)
    mask[c_edges == 255] = 1
    mask = ndi.binary_fill_holes(mask)

    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))

    # Done with imgs in range [0, 1]
    masked_img = (mask.astype('float32') * (image.astype('float32') / 255.0)) + ((1-mask.astype('float32')) * mask_color)
    masked_img = masked_img * 255

    return masked_img