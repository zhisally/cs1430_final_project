import cv2
import numpy as np
from imutils import contours
import segmentation
from color_kmeans import classify_colors, match_color
import kociemba as kc

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

def getKociembaString(cube_state_list):
    kociemba_string = ""
    for color in cube_state_list:
        kociemba_string = kociemba_string + kociemba[color]
    return kociemba_string


def main():

    cube_state = []

    # parser = argparse.ArgumentParser()

    # parser.add_argument('-i', '--images', help='File path to images folder')
    # args = parser.parse_args()
    # image_folder = args.images

    # for filename in sorted(os.listdir(image_folder)):
    #     if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
    #         path = os.path.join(image_folder, filename)
    #         face_state = detectColors(path, img_number)
    #         cube_state.extend(face_state)

    for i in range(6):
        pic = str(i + 1)
        image = cv2.imread('images/face' + pic + '.jpeg')
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
        
        print('Face ' + pic + ': ', color_lst)
    
    cube_state_string = getKociembaString(cube_state)
    if (len(cube_state_string) != 54):
        print("Kociemba string not appropriate length")
    print("Solution:")
    print(kc.solve(cube_state_string))
    
        
if __name__ == '__main__':
    main()

