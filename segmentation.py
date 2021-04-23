import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from imutils import rotate
import math

'''
Function removes background of image input
Takes image (array form) and outputs an image array of same dimensions
but with background removed
'''
def remove_background(image):
    # Canny Edge Detector 
    canny_low = 50 # Started at 15
    canny_high = 150
    
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    c_edges = cv2.Canny(gray_img, canny_low, canny_high)

    # Removes noise in background
    kernel = np.ones((20, 20), dtype=np.uint8) # Check effects of kernel size on output
    c_edges = cv2.morphologyEx(c_edges, cv2.MORPH_CLOSE, kernel)

    mask = np.zeros(c_edges.shape, dtype=np.uint8)
    mask[c_edges == 255] = 1
    mask = ndi.binary_fill_holes(mask)

    mask = np.expand_dims(mask, axis=2)
    mask = np.tile(mask, (1, 1, 3))
    # Done with imgs in range [0, 1]
    masked_img = mask * image
    return masked_img

'''
Given an image and a list of potential corners, find the four corners of the
Rubik's cube
params:
    img: image of cube
    dst: values indicating if a pixel in image is a potential corner
returns:
    4x2 array of the coordinates of the four corners
'''
def find_four_corners(img, dst):
    # get all coordinates of potential corners
    coords_img = []
    if not dst is None:
        coords_img = img.copy()
        coords_img[dst>0.01*dst.max()] = [0,0,254]
        coords_img[dst<=0.01*dst.max()] = [255,255,255]
        cv2.imshow('coords',coords_img)
    else:
        coords_img = img

    coords = np.where(coords_img[:,:,2] == 254)
    coords = np.array(coords).T

    # get top left corner
    dist = np.linalg.norm(coords-np.zeros((coords.shape[0], 2)), axis = 1)
    top_left = coords[np.argmin(dist), :]
    
    # get top right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    top_right = coords[np.argmin(dist), :]

    # get bottom left corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_left = coords[np.argmin(dist), :]

    # get bottom right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_right = coords[np.argmin(dist), :]
    
    return np.vstack((top_left, top_right, bottom_left, bottom_right)), coords_img

'''
Calculates the angle of rotation and then rotates the image.
params:
    img: image of cube
    corners: list of four corners of cube
returns:
    the rotated image
'''
def rotate_image(img, corners):
    top_left = corners[0,:]
    top_right = corners[1,:]
    
    # calculate angle
    angle = 0
    opp = top_right[0] - top_left[0]
    adj = top_right[1] - top_left[1]
    angle = np.arctan(opp/adj)*180/math.pi

    if (abs(angle) < 3):
        return img
    # rotate image
    rotated = rotate(img, angle)
    return rotated

'''
Crops the given image from the given corners.
params: 
    img: image of the cube
    corners: four corners of the cube
returns:
    the cropped image
'''
def crop_image(img, corners):
    print(corners.shape)
    top_left = corners[0,:]
    bottom_right = corners[3,:]
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    crop_img = img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width]
    return crop_img

'''
Performs Harris corner detection.
params: 
    img: the image of the cube
returns:
    img: img with detected corners marked out
    dst: values returned by cornerHarris
'''
def harris_corner(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,8,3,0.04)
    # result is dilated for marking the corners
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image
    img[dst>0.01*dst.max()]=[0,0,255]

    return img, dst

'''
Main corner detection helper. Uses Harris corner detection to detect all potential corners,
finds the four corners of the cube, rotates and crops the image.
params:
    image: image of the cube
returns:
    image of the straightened and cropped cube
'''
def corner_detection(image):
    original = image.copy()
    # blurs entire image
    image = cv2.GaussianBlur(image, (15, 15), 0)
    # detects corners using harris
    img, dst = harris_corner(image)

    # finds four corners of cube
    four_corners, coords_image = find_four_corners(img, dst)
    
    # rotate the image so cube is straight on
    rotated = rotate_image(original, four_corners)
    # rotate the corner coordinates to match the rotated image
    rotated_coords = rotate_image(coords_image, four_corners)
    
    
    rotated_original = rotated.copy()
    # find the corners in the newly rotated image
    four_corners, coords_image = find_four_corners(rotated_coords, None)

    # crop the image
    cropped_cube = crop_image(rotated_original, four_corners)

    return cropped_cube

'''
Gets a specified subsqare in the image
'''
def get_square(image, dim=3, row=0, col=0):
    h = image.shape[0] // dim
    w = image.shape[1] // dim
    return image[(row*h):((row+1)*h), (col*w):((col+1)*w), :]


