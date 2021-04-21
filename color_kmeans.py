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
    # cv2 defaults to reading images in BGR - need to convert to RGB first
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Want K-Means clustering to operate in 2-D, with one dimension the RGB values
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)

    # Set up K-Means model and applies it to fit the modified_image elements
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    # Count the number of points belonging to each cluster
    # Sets up dictionary for pie chart display
    counts = Counter(labels)
    counts = dict(sorted(counts.items(), key=lambda x: x[1]))

    # Obtain cluster centers
    center_colors = clf.cluster_centers_

    # Obtain colors by iterating through the labels in counts dictionary
    # Convert to hex for pie chart display
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    # Display pie chart is show_chart = True
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        plt.show()

    return rgb_colors



'''
match_color: Finds the best color string match out of the COLORS dictionary
    (all 6 possible colors for a Rubik's Cube) for the input list of RGB values

@params:
    rgb_colors: list of 1-D arrays, with each 1-D array containing [R, G, B] values 
    threshold: integer representing maximum difference from a string's color range. 
        To be matched with a string, the input color must be within threshold of 
        the string's color
@return:
    a string indicating the best matched color string out of the COLORS dictionary
        for the input RGB values.
'''
def match_color(rgb_colors, threshold=60):
    # Dictionary of 6 colors present on classic Rubik's Cube
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

    # Loop through RGB colors
    for i in range(len(rgb_colors)):
        # Convert current rgb_color to CIE 1976 Lab values for comparison 
        curr_color = rgb2lab(np.uint8(np.asarray([[rgb_colors[i]]])))
        # Loop through COLORS dictionary
        for color, rgb_vals in COLORS.items():
            # Also convert current color from dictionary into Lab values
            selected_color = rgb2lab(np.uint8(np.asarray([[rgb_vals]])))
            # Compute differences between input color and dictionary color
            diff = deltaE_cie76(selected_color, curr_color)
            # If difference is less than threshold and is less than the current minimum
            if diff < threshold and diff < min_diff:
                # Set current matched color to dictionary color
                matched_color = color
                # Set minimum to the new minimum difference 
                min_diff = diff
    
    return matched_color