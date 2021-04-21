# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import ndimage as ndi
# from imutils import rotate
# import math

# '''
# 4/10/21: Function removes background of image input
# Takes image (array form) and outputs an image array of same dimensions
# but with background removed
# '''
# def remove_background(image):
#     # Canny Edge Detector (check if need adjust depending on image)
#     canny_low = 50 # Started at 15
#     canny_high = 150

#     gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     c_edges = cv2.Canny(gray_img, canny_low, canny_high)

#     # Removes noise in background
#     kernel = np.ones((20, 20), dtype=np.uint8) # Check effects of kernel size on output
#     c_edges = cv2.morphologyEx(c_edges, cv2.MORPH_CLOSE, kernel)

#     mask = np.zeros(c_edges.shape, dtype=np.uint8)
#     mask[c_edges == 255] = 1
#     mask = ndi.binary_fill_holes(mask)

#     mask = np.expand_dims(mask, axis=2)
#     mask = np.tile(mask, (1, 1, 3))

#     # Done with imgs in range [0, 1]
#     masked_img = mask * image

#     return masked_img

# def find_four_corners(img, dst):
#     # get all points
#     coords_img = img.copy()
#     coords_img[dst>0.01*dst.max()] = [0,0,0]
#     coords_img[dst<=0.01*dst.max()] = [255,255,255]
#     cv2.imshow('coords',coords_img)

#     # get the coords of each point
#     coords = np.where(coords_img[:,:,0] == 0)
#     coords = np.array(coords).T
#     # print(coords.shape)

#     # get top left corner
#     dist = np.linalg.norm(coords-np.zeros((coords.shape[0], 2)), axis = 1)
#     top_left = coords[np.argmin(dist), :]
#     # print(top_left)
    
#     # get top right corner
#     corner = np.zeros((coords.shape[0], 2))
#     corner[:,1] = img.shape[1]
#     dist = np.linalg.norm(coords-corner, axis = 1)
#     top_right = coords[np.argmin(dist), :]
#     # print(top_right)

#     # get bottom left corner
#     corner = np.zeros((coords.shape[0], 2))
#     corner[:,0] = img.shape[0]
#     dist = np.linalg.norm(coords-corner, axis = 1)
#     bottom_left = coords[np.argmin(dist), :]
#     # print(bottom_left)

#     # get bottom right corner
#     corner = np.zeros((coords.shape[0], 2))
#     corner[:,0] = img.shape[0]
#     corner[:,1] = img.shape[1]
#     dist = np.linalg.norm(coords-corner, axis = 1)
#     bottom_right = coords[np.argmin(dist), :]
#     # print(bottom_right)
#     return np.vstack((top_left, top_right, bottom_left, bottom_right))

# def rotate_image(img, corners):
#     top_left = corners[0,:]
#     top_right = corners[1,:]
    
#     angle = 0
#     opp = top_right[0] - top_left[0]
#     adj = top_right[1] - top_left[1]
#     angle = np.arctan(opp/adj)*180/math.pi

#     if (abs(angle) < 2):
#         return img
#     # print('here')
#     rotated = rotate(img, angle)
#     return rotated

# def crop_image(img, corners):
#     top_left = corners[0,:]
#     bottom_right = corners[3,:]
#     width = bottom_right[1] - top_left[1]
#     height = bottom_right[0] - top_left[0]
#     crop_img = img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width]
#     return crop_img

# def harris_corner(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
#     gray = np.float32(gray)
#     dst = cv2.cornerHarris(gray,5,3,0.04)
#     # result is dilated for marking the corners, not important
#     dst = cv2.dilate(dst,None)
#     # print(dst.shape)
#     # Threshold for an optimal value, it may vary depending on the image.
#     img[dst>0.01*dst.max()]=[0,0,255]

#     return img, dst

# '''
# Main corner detection helper
# Returns the cropped image
# '''
# def corner_detection(image):
#     original = image.copy()
#     img, dst = harris_corner(image)

#     cv2.imshow("img", img)
#     four_corners = find_four_corners(img, dst)
    
#     rotated = rotate_image(original, four_corners)
#     cv2.imshow("rotated", rotated)
    
#     rotated_original = rotated.copy()
#     rotated, rotated_dst = harris_corner(rotated)
#     four_corners = find_four_corners(rotated, rotated_dst)

#     cropped_cube = crop_image(rotated_original, four_corners)

#     cv2.imshow("cropped", cropped_cube)

#     return cropped_cube

# def get_square(image, dim=3, row=0, col=0):
#     h = image.shape[0] // dim
#     w = image.shape[1] // dim
#     return image[(row*h):((row+1)*h), (col*w):((col+1)*w), :]


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from imutils import rotate
import math

'''
4/10/21: Function removes background of image input
Takes image (array form) and outputs an image array of same dimensions
but with background removed
'''
def remove_background(image):
    # Canny Edge Detector (check if need adjust depending on image)
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
    return mask, masked_img

def find_four_corners(img, dst):
    # get all points
    coords_img = []
    if not dst is None:
        coords_img = img.copy()
        coords_img[dst>0.01*dst.max()] = [0,0,254]
        coords_img[dst<=0.01*dst.max()] = [255,255,255]
        cv2.imshow('coords',coords_img)
    else:
        coords_img = img

    print(coords_img.shape)
    # get the coords of each point
    coords = np.where(coords_img[:,:,2] == 254)
    coords = np.array(coords).T
    # print(coords.shape)
    # print(coords)
    # get top left corner
    dist = np.linalg.norm(coords-np.zeros((coords.shape[0], 2)), axis = 1)
    top_left = coords[np.argmin(dist), :]
    # print(top_left)
    
    # get top right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    top_right = coords[np.argmin(dist), :]
    # print(top_right)

    # get bottom left corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_left = coords[np.argmin(dist), :]
    # print(bottom_left)

    # get bottom right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_right = coords[np.argmin(dist), :]
    # print(bottom_right)
    # print(np.vstack((top_left, top_right, bottom_left, bottom_right)))
    return np.vstack((top_left, top_right, bottom_left, bottom_right)), coords_img

def rotate_image(img, corners):
    top_left = corners[0,:]
    top_right = corners[1,:]
    
    angle = 0
    opp = top_right[0] - top_left[0]
    adj = top_right[1] - top_left[1]
    angle = np.arctan(opp/adj)*180/math.pi

    if (abs(angle) < 3):
        return img
    # print('here')
    rotated = rotate(img, angle)
    return rotated

def crop_image(img, corners):
    print(corners.shape)
    top_left = corners[0,:]
    bottom_right = corners[3,:]
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    crop_img = img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width]
    return crop_img

def harris_corner(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,8,3,0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # print(dst.shape)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    return img, dst

'''
Main corner detection helper
Returns the cropped image
'''
def corner_detection(image):
    original = image.copy()
    image = cv2.GaussianBlur(image, (15, 15), 0)
    # kernel = np.ones((21,21),np.uint8)
    # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    img, dst = harris_corner(image)

    cv2.imshow("img", img)
    four_corners, coords_image = find_four_corners(img, dst)
    
    print(four_corners.shape)
    rotated = rotate_image(original, four_corners)
    cv2.imshow("rotated", rotated)

    rotated_coords = rotate_image(coords_image, four_corners)
    cv2.imshow("rotated coords", rotated_coords)
    
    rotated_original = rotated.copy()
    four_corners, coords_image = find_four_corners(rotated_coords, None)
    # rotated, rotated_dst = harris_corner(rotated)
    # four_corners,_ = find_four_corners(rotated, rotated_dst)
    
    cropped_cube = crop_image(rotated_original, four_corners)

    cv2.imshow("cropped", cropped_cube)
    return cropped_cube

def get_square(image, dim=3, row=0, col=0):
    h = image.shape[0] // dim
    w = image.shape[1] // dim
    return image[(row*h):((row+1)*h), (col*w):((col+1)*w), :]



def detect_cube(img):
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)


    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]


    ## This will extract the rotated rect from the contour
    rot_rect = cv2.minAreaRect(cnt)

    # Extract useful data
    cx,cy = (rot_rect[0][0], rot_rect[0][1]) # rect center
    sx,sy = (rot_rect[1][0], rot_rect[1][1]) # rect size
    angle = rot_rect[2] # rect angle


    # Set model points : The original shape
    model_pts = np.array([[0,sy],[0,0],[sx,0],[sx,sy]]).astype('int')
    # Set detected points : Points on the image
    current_pts = cv2.boxPoints(rot_rect).astype('int')

    # sort the points to ensure match between model points and current points
    ind_model = np.lexsort((model_pts[:,1],model_pts[:,0]))
    ind_current = np.lexsort((current_pts[:,1],current_pts[:,0]))

    model_pts = np.array([model_pts[i] for i in ind_model])
    current_pts = np.array([current_pts[i] for i in ind_current])


    # Estimate the transform betwee points
    M = cv2.estimateRigidTransform(current_pts,model_pts,True)

    # Wrap the image
    wrap_gray = cv2.warpAffine(img, M, (int(sx),int(sy)))
    return wrap_gray
