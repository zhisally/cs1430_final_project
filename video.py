import cv2
import numpy as np
from imutils import contours

colors = {
    'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
    'blue': ([69, 120, 100], [179, 255, 255]),    # Blue
    'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
    'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
    'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
    'white' : ([0,0,0], [0,0,255]) # White
    # 'red' :
    }

faces = ["front", "right", "back", "left", "top", "bottom"]
face_index = 0
webcam = cv2.VideoCapture(0)
while True:
    _, image = webcam.read()
    key = cv2.waitKey(10) & 0xFF
        
    # image = cv2.imread('4x4.png')
    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape, dtype=np.uint8)

    # Color threshold to find the squares
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        color_mask = cv2.inRange(image, lower, upper)
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
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

            cv2.putText(image, "#{}".format(number + 1), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            number += 1
        
    if key == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break

    if key == ord('s'):
        # cv2.putText(image, "Press s to capture face", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        face_index += 1
        if face_index == 6:
            webcam.release()
            cv2.destroyAllWindows()
            break
    else:
        cv2.putText(image, "Press s to capture " + faces[face_index] + " face", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

    cv2.imshow("Real Time Color Detection", image)
        
        # loop through contours
        # save colors into array
        # keep count of how many times s is pressed (only press 6 times)
        # after the 6th time, break out of the loop and rearrange array, apply solver algorithm


    # cv2.imshow('mask', mask)
    # cv2.imwrite('mask.png', mask)
    # cv2.imshow('original', original)
    # cv2.waitKey()