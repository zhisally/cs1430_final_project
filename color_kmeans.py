from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

'''
rgb_to_hex: converts RGB values into hexadecimal form suitable for pie chart display

@params:
    color: 1-D array containing RGB color values as [Red, Green, Blue]
@return:
    The hexadecimal color code of the RGB values
'''
def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))



'''
classify_colors: Performs K-Means clustering on colors in input image

@params:
    image: 3-D array (third dimension are the R, G, B channels) image to 
        use for K-Means clustering
    number_of_colors: integer number of colors we want K-Means to detect (corresponds
        to the number of clusters we pass into K-Means)
    show_chart: Boolean indicating whether to display a piechart showing the
        percentage of points classified within each cluster and the centroid color
@return:
    a list of 1-D arrays, each the [R, G, B] values for each cluster center after
        K-Means clustering of the image's colors 
'''
def classify_colors(image, number_of_colors, show_chart=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    center_colors = clf.cluster_centers_

    # Obtain ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.show()

    return rgb_colors

def match_color(rgb_colors, threshold=60):
    COLORS = {
        'ORANGE': [255, 140, 0],
        'RED': [255, 0, 0],
        'GREEN': [0, 128, 0],
        'BLUE': [0, 0, 128],
        'YELLOW': [255, 255, 0],
        'WHITE': [255, 255, 255]
        }

    min_diff = threshold
    matched_color = None

    for i in range(len(rgb_colors)):
        curr_color = rgb2lab(np.uint8(np.asarray([[rgb_colors[i]]])))
        for color, rgb_vals in COLORS.items():
            selected_color = rgb2lab(np.uint8(np.asarray([[rgb_vals]])))
            diff = deltaE_cie76(selected_color, curr_color)
            if diff < threshold and diff < min_diff:
                matched_color = color
                min_diff = diff
    
    return matched_color