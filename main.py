import cv2
import numpy as np
from imutils import contours
from segmentation import remove_background

def checkInRange(colors, curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key

def main():
    image = cv2.imread('3x3.jpeg')
    
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
        if i % 3 == 0:  
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
    cv2.rectangle(original, (415, 0), (415 + 50, 0 + 50), (36,255,12), 2)
    cv2.imshow('mask', mask)
    cv2.imwrite('mask.png', mask)
    cv2.imshow('original', original)
    key = cv2.waitKey() & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    main()

