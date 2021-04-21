import numpy as np
from imutils import contours
from segmentation import remove_background
import kociemba as kc
import argparse
import os
import cv2

cube_state = []

kociemba = {
        'blue': 'F',
        'yellow': 'L',
        'orange': 'D',
        'green' : 'B',
        'red' : 'U',
        'red2' : 'U',
        'white': 'R'
        }

def checkInRange(colors, curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key
    print("not in any color range")

def edgeDetection():
    imgobj = cv2.imread('4x4/f1.png')
    gray = cv2.cvtColor(imgobj, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("image")
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blurred, 20, 40)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=2)

    cv2.imshow('image', dilated)

    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    hierarchy = hierarchy[0]

    index = 0
    pre_cX = 0
    pre_cY = 0
    center = []
    for component in zip(contours, hierarchy):
        contour = component[0]
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.1 * peri, True)
        area = cv2.contourArea(contour)
        corners = len(approx)

        # compute the center of the contour
        M = cv2.moments(contour)

        if M["m00"]:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = None
            cY = None

        if 14000 < area < 20000 and cX is not None:
            tmp = {'index': index, 'cx': cX, 'cy': cY, 'contour': contour}
            center.append(tmp)
            index += 1

    center.sort(key=lambda k: (k.get('cy', 0)))
    row1 = center[0:3]
    row1.sort(key=lambda k: (k.get('cx', 0)))
    row2 = center[3:6]
    row2.sort(key=lambda k: (k.get('cx', 0)))
    row3 = center[6:9]
    row3.sort(key=lambda k: (k.get('cx', 0)))

    center.clear()
    center = row1 + row2 + row3

    print(center)
    for component in center:
        candidates.append(component.get('contour'))

    cv2.drawContours(imgobj, candidates, -1, (0, 0, 255), 3)
    cv2.imshow("final", imgobj)
    cv2.waitKey(0)

def sobel(image):
    sobel_64 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    abs_64 = np.absolute(sobel_64)
    sobel_8u = np.uint8(abs_64)
    cv2.imshow('sobel', sobel_8u)


def detectColors(file, number):
    image = cv2.imread(file)
    original = image.copy()

    image = remove_background(image)
    masked_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape, dtype=np.uint8)

    colors = {
        'blue': ([90, 50, 70], [128, 255, 255]),
        'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
        'red' : ([0, 100, 100], [8, 255, 255]), #Red,
        'red2' : ([159, 50, 70], [180, 255, 255]), #Red
        'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
        'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
        'white': ([0,0,210], [255,255,255])
        }

    # Color threshold to find the squares
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(masked_img, lower, upper)

        #remove noise
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)

        color_mask = cv2.merge([color_mask, color_mask, color_mask])
        mask = cv2.bitwise_or(mask, color_mask)

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Sort all contours from top-to-bottom or bottom-to-top
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    img_contours = np.zeros(image.shape)
    cv2.drawContours(img_contours, cnts, -1, (0,255,0), 3)
    cv2.imshow("contours" + str(number), img_contours)

    new_cnts = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        if (w*h) >= 4000:
            new_cnts.append(contour)

    
    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []

    for (i, c) in enumerate(new_cnts, 1):
        row.append(c)
        if i % 3 == 0:  
            (new_cnts, _) = contours.sort_contours(row, method="left-to-right")
            cube_rows.append(new_cnts)
            row = []

    face_colors = []
    # Draw text
    face_num = 1
    for row in cube_rows:
        for c in row:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
            curr_color = ""
            curr_x = x + w//8
            curr_y = y + h//8
            if (curr_x >= 0 and curr_x <= masked_img.shape[1] and curr_y >= 0 and curr_y <= masked_img.shape[0]):
                curr_color = checkInRange(colors, masked_img[curr_y][curr_x])
            string = str(curr_color ) + " " + str("#{}".format(face_num))
            cv2.putText(original, string, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            face_colors.append(curr_color)
            face_num += 1

    cv2.imshow('mask' + str(number), mask)
    cv2.imshow('photo with bounding squares ' + str(number), original)
    
    return face_colors

def getKociembaString(cube_state_list):
    kociemba_string = ""
    for color in cube_state_list:
        kociemba_string = kociemba_string + kociemba[color]
    return kociemba_string

def main():
    cube_state.clear()

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', help='File path to images folder')
    args = parser.parse_args()
    image_folder = args.images
    
    img_number = 1

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(image_folder, filename)
            face_state = detectColors(path, img_number)
            cube_state.extend(face_state)
            img_number += 1
    kociemba_string = getKociembaString(cube_state)
    
    key = cv2.waitKey() & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
    if (len(kociemba_string) != 54):
        print("Incorrect length for Kociemba input")
    print(kc.solve('RRBBUFBFBRLRRRFRDDURUBFBBRFLUDUDFLLFFLLLLDFBDDDUUBDLUU'))

if __name__ == '__main__':
    main()