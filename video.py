import cv2
import numpy as np
from imutils import contours
import segmentation
import color_kmeans

'''
Check if pixel is in any of the color ranges
'''
def checkInRange(colors, curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key
    print("not in any color range")

# color ranges
colors = {
        'blue': ([90, 50, 70], [128, 255, 255]), # Blue
        'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
        'red' : ([0, 100, 100], [8, 255, 255]), #Red,
        'red2' : ([159, 50, 70], [180, 255, 255]), #Red
        'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
        'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
        'white': ([0,0,210], [255,255,255]) # White
        }


# define webcam
webcam = cv2.VideoCapture(0)
cube_state = []
while True:
    # get frame
    _, image = webcam.read()
    original = image.copy()
    key = cv2.waitKey(10) & 0xFF

    # define bounding box in which to place cube
    box_start = (image.shape[1] // 2 - image.shape[1] // 6, image.shape[0] // 2 - image.shape[1] // 6)
    cv2.rectangle(original, box_start, (box_start[0] + image.shape[1] // 3, box_start[1] + image.shape[1] // 3), (36,255,12), 2)     
    cropped_cube = image[box_start[1]:box_start[1]+image.shape[1] // 3, box_start[0]:box_start[0] + image.shape[1] // 3]
    cropped_cube = segmentation.remove_background(cropped_cube)
    cropped_cube = cv2.cvtColor(cropped_cube, cv2.COLOR_BGR2HSV)
    mask = np.zeros(cropped_cube.shape, dtype=np.uint8)

    # Color threshold to find the squares
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(cropped_cube, lower, upper)

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

    # remove contours below certain size limit
    new_cnts = []
    for contour in cnts:
        x, y, w, h = cv2.boundingRect(contour)
        if (w*h) >= 4000:
            new_cnts.append(contour)

    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []

    for (i, c) in enumerate(cnts, 1):
        x,y,w,h = cv2.boundingRect(c)
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
                cv2.rectangle(original, (x + box_start[0], y + box_start[1]), (x + box_start[0] + w, y + box_start[1] + h), (36,255,12), 2)
                curr_color = ""
                curr_x = x + w//2
                curr_y = y + h//2

                if (curr_x >= 0 and curr_x <= image.shape[1] and curr_y >= 0 and curr_y <= image.shape[0]):
                    curr_color = checkInRange(colors, image[curr_y][curr_x])
                string = str(curr_color ) + " " + str("#{}".format(number + 1))
                cv2.putText(original, string, (x+box_start[0],y+box_start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                number += 1

    # press q to exit webcam
    if key == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow("Real Time Color Detection", original)
    