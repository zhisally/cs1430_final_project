import numpy as np
import cv2
from imutils import rotate
from segmentation import remove_background
import math

def find_four_corners(img, dst):
    # get all points
    coords_img = img.copy()
    coords_img[dst>0.01*dst.max()] = [0,0,0]
    coords_img[dst<=0.01*dst.max()] = [255,255,255]
    cv2.imshow('coords',coords_img)

    # get the coords of each point
    coords = np.where(coords_img[:,:,0] == 0)
    coords = np.array(coords).T
    print(coords.shape)

    # get top left corner
    dist = np.linalg.norm(coords-np.zeros((coords.shape[0], 2)), axis = 1)
    top_left = coords[np.argmin(dist), :]
    print(top_left)
    
    # get top right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    top_right = coords[np.argmin(dist), :]
    print(top_right)

    # get bottom left corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_left = coords[np.argmin(dist), :]
    print(bottom_left)


    # get bottom right corner
    corner = np.zeros((coords.shape[0], 2))
    corner[:,0] = img.shape[0]
    corner[:,1] = img.shape[1]
    dist = np.linalg.norm(coords-corner, axis = 1)
    bottom_right = coords[np.argmin(dist), :]
    print(bottom_right)
    return np.vstack((top_left, top_right, bottom_left, bottom_right))

def rotate_image(img, corners):
    top_left = corners[0,:]
    top_right = corners[1,:]
    
    angle = 0
    opp = top_right[0] - top_left[0]
    adj = top_right[1] - top_left[1]
    angle = np.arctan(opp/adj)*180/math.pi

    if (abs(angle) < 2):
        return img
    # print('here')
    rotated = rotate(img, angle)
    return rotated

def crop_image(img, corners):
    top_left = corners[0,:]
    bottom_right = corners[3,:]
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    crop_img = img[top_left[0]:top_left[0]+height, top_left[1]:top_left[1]+width]
    return crop_img

def harris_corner(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,5,3,0.04)
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    print(dst.shape)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    return img, dst

def corner_detection(image):
    img = cv2.imread(image)
    # img = remove_background(img)
    # cv2.imshow("removed background", img)
    original = img.copy()
    img, dst = harris_corner(img)

    cv2.imshow("img", img)
    four_corners = find_four_corners(img, dst)
    
    rotated = rotate_image(original, four_corners)
    cv2.imshow("rotated", rotated)
    
    rotated_original = rotated.copy()
    rotated, rotated_dst = harris_corner(rotated)
    four_corners = find_four_corners(rotated, rotated_dst)

    cropped_cube = crop_image(rotated_original, four_corners)

    cv2.imshow("cropped", cropped_cube)

    key = cv2.waitKey() & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()

def main():
    corner_detection("better_lighting/face6.jpeg")
    # corner_detection("f1.png")
        

if __name__ == '__main__':
    main()
