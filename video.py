# import cv2
# import numpy as np
# from imutils import contours
# from segmentation import remove_background
# from main import checkInRange

# # colors = {
#     # 'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
#     # 'blue': ([69, 120, 100], [179, 255, 255]),    # Blue
#     # 'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
#     # 'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
#     # 'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
#     # 'white' : ([0,0,0], [0,0,255]) # White
#     # # 'red' :
#     # }

# colors = {
#         # 'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
#         'blue': ([90, 50, 70], [130, 255, 255]),
#         'yellow': ([20, 100, 100], [40, 255, 255]),   # Yellow
#         'orange': ([0, 50, 50], [30, 255, 255]),     # Orange
#         'green' : ([60 - 14, 100, 100], [60 + 20, 255, 255]), # Green
#         'red' : ([159, 50, 70], [180, 255, 255]), #Red
#         'white' : ([0,0,230], [255,255,255]) # White
#         }

# faces = ["front", "right", "back", "left", "top", "bottom"]
# face_index = 0
# webcam = cv2.VideoCapture(0)
# while True:
#     _, image = webcam.read()
#     key = cv2.waitKey(10) & 0xFF
        
#     original = image.copy()

#     mask,image = remove_background(image)

#     # cv2.imshow('background_removed', image)

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     # cv2.imshow('hsv', image)
#     # print(image[176, 774])
#     mask = np.zeros(image.shape, dtype=np.uint8)

#     # Color threshold to find the squares
#     open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
#     close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
#     for color, (lower, upper) in colors.items():
#         lower = np.array(lower, dtype=np.uint8)
#         upper = np.array(upper, dtype=np.uint8)
#         color_mask = cv2.inRange(image, lower, upper)

#         #remove noise
#         color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
#         color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)

#         color_mask = cv2.merge([color_mask, color_mask, color_mask])
#         mask = cv2.bitwise_or(mask, color_mask)

#     gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     # Sort all contours from top-to-bottom or bottom-to-top
#     (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

#     # Take each row of 3 and sort from left-to-right or right-to-left
#     cube_rows = []
#     row = []

#     for (i, c) in enumerate(cnts, 1):
#         row.append(c)
#         if i % 3 == 0:  
#             (cnts, _) = contours.sort_contours(row, method="left-to-right")
#             cube_rows.append(cnts)
#             row = []

#     # Draw text
#     number = 0
#     for row in cube_rows:
#         for c in row:
#             x,y,w,h = cv2.boundingRect(c)
#             if (w*h >= 2000):
#                 cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
#                 curr_color = ""
#                 curr_x = x + w//2
#                 curr_y = y + h//2
#                 # -----testing stuff ----
#                 # print("x: ", curr_x)
#                 # print("y: ", curr_y)
#                 # find average pixel value of rectangle
#                 # curr = np.mean(image[curr_x-10:curr_x+10][curr_y-10:curr_y+10], axis = ((0,1)))
#                 # print(curr.shape)
#                 # -----testing stuff ----

#                 if (curr_x >= 0 and curr_x <= image.shape[1] and curr_y >= 0 and curr_y <= image.shape[0]):
#                     curr_color = checkInRange(colors, image[curr_y][curr_x])
#                     # curr_color = checkInRange(colors, curr)
#                 string = str(curr_color ) + " " + str("#{}".format(number + 1))
#                 cv2.putText(original, string, (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
#                 number += 1
        
#     if key == ord('q'):
#         webcam.release()
#         cv2.destroyAllWindows()
#         break

#     if key == ord('s'):
#         # cv2.putText(image, "Press s to capture face", (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
#         face_index += 1
#         if face_index == 6:
#             webcam.release()
#             cv2.destroyAllWindows()
#             break
#     else:
#         cv2.putText(original, "Press s to capture " + faces[face_index] + " face", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

#     cv2.imshow("Real Time Color Detection", original)
        
#         # loop through contours
#         # save colors into array
#         # keep count of how many times s is pressed (only press 6 times)
#         # after the 6th time, break out of the loop and rearrange array, apply solver algorithm


#     # cv2.imshow('mask', mask)
#     # cv2.imwrite('mask.png', mask)
#     # cv2.imshow('original', original)
#     # cv2.waitKey()











import cv2
import numpy as np
from imutils import contours
import segmentation
from main import checkInRange
import color_kmeans

# colors = {
    # 'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
    # 'blue': ([69, 120, 100], [179, 255, 255]),    # Blue
    # 'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
    # 'orange': ([0, 110, 125], [17, 255, 255]),     # Orange
    # 'green' : ([60 - 20, 100, 100], [60 + 20, 255, 255]), # Green
    # 'white' : ([0,0,0], [0,0,255]) # White
    # # 'red' :
    # }

colors = {
        # 'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
        'blue': ([90, 50, 70], [130, 255, 255]),
        'yellow': ([20, 100, 100], [40, 255, 255]),   # Yellow
        'orange': ([0, 50, 50], [30, 255, 255]),     # Orange
        'green' : ([60 - 14, 100, 100], [60 + 20, 255, 255]), # Green
        'red' : ([159, 50, 70], [180, 255, 255]), #Red
        'white' : ([0,0,230], [255,255,255]) # White
        }

faces = ["front", "right", "back", "left", "top", "bottom"]
face_index = 0
webcam = cv2.VideoCapture(0)
cube_state = []
while True:
    _, image = webcam.read()
    original = image.copy()
    key = cv2.waitKey(10) & 0xFF

    box_start = (image.shape[1] // 2 - image.shape[1] // 6, image.shape[0] // 2 - image.shape[1] // 6)

    cv2.rectangle(original, box_start, (box_start[0] + image.shape[1] // 3, box_start[1] + image.shape[1] // 3), (36,255,12), 2)
        
    cropped_cube = image[box_start[1]:box_start[1]+image.shape[1] // 3, box_start[0]:box_start[0] + image.shape[1] // 3]

    # mask,image = segmentation.remove_background(image)

    cropped_cube = cv2.cvtColor(cropped_cube, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', image)
    # print(image[176, 774])
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

    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []

    for (i, c) in enumerate(cnts, 1):
        x,y,w,h = cv2.boundingRect(c)
        # if (w*h >= 2000):
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
                cv2.putText(original, string, (x+box_start[0],y+box_start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                number += 1




    # kmeans
    # dims = 3
    # color_lst = []
    # for r in range(dims):
    #     for c in range(dims):
    #         square = segmentation.get_square(cropped_cube, dim=dims, row=r, col=c)
    #         rgb_colors = color_kmeans.classify_colors(square, 2, show_chart=False)
    #         matched_color = color_kmeans.match_color(rgb_colors)
    #         h = cropped_cube.shape[0] // dims
    #         w = cropped_cube.shape[1] // dims
    #         cv2.putText(original, matched_color, (box_start[0] + c*w,box_start[1] + r*h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    #         color_lst.append(matched_color)
    #         cube_state.append(matched_color)

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
        cv2.putText(original, "Press s to capture " + faces[face_index] + " face", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

    cv2.imshow("Real Time Color Detection", original)
        
        # loop through contours
        # save colors into array
        # keep count of how many times s is pressed (only press 6 times)
        # after the 6th time, break out of the loop and rearrange array, apply solver algorithm


    # cv2.imshow('mask', mask)
    # cv2.imwrite('mask.png', mask)
    # cv2.imshow('original', original)
    # cv2.waitKey()