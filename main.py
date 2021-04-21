import cv2
import numpy as np
from imutils import contours
import segmentation
from color_kmeans import classify_colors, match_color
import kociemba as kc
import argparse
import os

kociemba = {
        'BLUE': 'F',
        'YELLOW': 'L',
        'ORANGE': 'D',
        'GREEN' : 'B',
        'RED' : 'U',
        'WHITE': 'R'
        }

def checkInRange(colors, curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key
    print("not in any color range")

def main():
    image = cv2.imread('f3.png')
    
    original = image.copy()

    image = remove_background(image)

    cv2.imshow('background_removed', image)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', image)
    mask = np.zeros(image.shape, dtype=np.uint8)

    colors = {
        # 'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
        'blue': ([90, 50, 70], [130, 255, 255]),
        'yellow': ([20, 100, 100], [40, 255, 255]),   # Yellow
        'orange': ([0, 50, 50], [30, 255, 255]),     # Orange
        'green' : ([60 - 14, 100, 100], [60 + 20, 255, 255]), # Green
        'red' : ([159, 50, 70], [180, 255, 255]), #Red
        'white' : ([0,0,230], [255,255,255]) # White
        }

    # Color threshold to find the squares
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(image, lower, upper)

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

    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []

    for (i, c) in enumerate(cnts, 1):
        row.append(c)
        if i % 4 == 0:  
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            cube_rows.append(cnts)
            row = []

    # Draw text
    number = 0
    for row in cube_rows:
        for c in row:
            x,y,w,h = cv2.boundingRect(c)
            if (w*h >= 2000):
                cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                curr_color = ""
                curr_x = x + w//2
                curr_y = y + h//2
                # -----testing stuff ----
                # print("x: ", curr_x)
                # print("y: ", curr_y)
                # find average pixel value of rectangle
                # curr = np.mean(image[curr_x-10:curr_x+10][curr_y-10:curr_y+10], axis = ((0,1)))
                # print(curr.shape)
                # -----testing stuff ----

                if (curr_x >= 0 and curr_x <= image.shape[1] and curr_y >= 0 and curr_y <= image.shape[0]):
                    curr_color = checkInRange(colors, image[curr_y][curr_x])
                    # curr_color = checkInRange(colors, curr)
                string = str(curr_color ) + " " + str("#{}".format(number + 1))
                cv2.putText(original, string, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                number += 1
    # cv2.rectangle(original, (415, 0), (415 + 50, 0 + 50), (36,255,12), 2)
    cv2.imshow('mask', mask)
    cv2.imwrite('mask.png', mask)
    cv2.imshow('original', original)
    key = cv2.waitKey() & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        
def getKociembaString(cube_state_list):
    kociemba_string = ""
    for color in cube_state_list:
        kociemba_string = kociemba_string + kociemba[color]
    return kociemba_string


def main():

    cube_state = []

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--images', help='File path to images folder')
    args = parser.parse_args()
    image_folder = args.images

    face_number = 1

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(image_folder, filename)
            image = cv2.imread(path)
            cropped_cube = segmentation.corner_detection(image)
            dims = 3
            color_lst = []

            for r in range(dims):
                for c in range(dims):
                    square = segmentation.get_square(cropped_cube, dim=dims, row=r, col=c)
                    rgb_colors = classify_colors(square, 2, show_chart=False)
                    matched_color = match_color(rgb_colors)
                    color_lst.append(matched_color)
                    cube_state.append(matched_color)
            print('Face ' + str(face_number) + ': ', color_lst)
            face_number += 1
    
    cube_state_string = getKociembaString(cube_state)
    if (len(cube_state_string) != 54):
        print("Kociemba string not appropriate length")
    print("Solution:")
    print(kc.solve(cube_state_string))
    
        
if __name__ == '__main__':
    main()

