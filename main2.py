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

colors = {
        'blue': ([90, 50, 70], [128, 255, 255]),
        'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
        'red' : ([0, 100, 100], [8, 255, 255]), #Red,
        'red2' : ([159, 50, 70], [180, 255, 255]), #Red
        'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
        'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
        'white': ([0,0,210], [255,255,255])
        }

'''
Checks whether the color is within the color ranges declared
'''
def checkInRange(curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key
    print("not in any color range")

'''
Takes in a file and file number in the folder and detects the colors of the cube face.
@param 
    file: file to image
    number: the number of the image in the sequence
@returns
    list of colors
'''
def detectColors(file, number):
    image = cv2.imread(file)
    original = image.copy()

    image = remove_background(image)
    masked_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape, dtype=np.uint8)

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
                curr_color = checkInRange(masked_img[curr_y][curr_x])
            string = str(curr_color ) + " " + str("#{}".format(face_num))
            cv2.putText(original, string, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            face_colors.append(curr_color)
            face_num += 1

    cv2.imshow('mask' + str(number), mask)
    cv2.imshow('photo with bounding squares ' + str(number), original)
    
    return face_colors

'''
Converts cube state list into kociemba format string
@params
    cube_state_list: list of cube colors for each face
@return
    kociemba string
'''
def getKociembaString(cube_state_list):
    kociemba_string = ""
    for color in cube_state_list:
        kociemba_string = kociemba_string + kociemba[color]
    return kociemba_string

def main():
    cube_state.clear()

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', default=os.getcwd() + '/images-main2/', help='File path to images folder')
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
