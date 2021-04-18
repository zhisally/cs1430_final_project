import cv2
import numpy as np
from imutils import contours
import segmentation
from color_kmeans import classify_colors, match_color

def checkInRange(colors, curr_pixel):
    for key, value in colors.items():
        lower_bound = value[0]
        upper_bound = value[1]
        if (curr_pixel[0] >= lower_bound[0] and curr_pixel[0] <= upper_bound[0] and curr_pixel[1] >= lower_bound[1] and curr_pixel[1] <= upper_bound[1] and  curr_pixel[2] >= lower_bound[2] and curr_pixel[2] <= upper_bound[2]):
            return key
    print("not in any color range")


def main():
    for i in range(6):
        pic = str(i + 1)
        image = cv2.imread('better_lighting/face' + pic + '.jpeg')
        cropped_cube = segmentation.corner_detection(image)
        dims = 3
        color_lst = []

        for r in range(dims):
            for c in range(dims):
                square = segmentation.get_square(cropped_cube, dim=dims, row=r, col=c)
                rgb_colors = classify_colors(square, 2, show_chart=False)
                matched_color = match_color(rgb_colors)
                color_lst.append(matched_color)
        
        print('Face ' + pic + ': ', color_lst)
        
if __name__ == '__main__':
    main()

