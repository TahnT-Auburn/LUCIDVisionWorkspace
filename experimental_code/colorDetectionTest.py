#%%
import cv2 as cv
import sys
import numpy as np

# define an image using imread
img = cv.imread(cv.samples.findFile("images/Red.jpg"))

# check if image is loaded correctly
if img is None:
    sys.exit("Could not read the image")

# ----- color detection -----
# BGR colors
white = np.uint8([[[255, 255, 255]]])
blue = np.uint8([[[255, 0, 0]]])
green = np.uint8([[[0, 255, 0]]])
red = np.uint8([[[0, 0, 255]]])

# HSV colors
hsv_white = cv.cvtColor(white, cv.COLOR_BGR2HSV)
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
hsv_red =  cv.cvtColor(red, cv.COLOR_BGR2HSV)

# select color for detection
color_bgr = hsv_red

# convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# generate HSV bounds
lower_bound = []
upper_bound = []

low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])


k = 0
while k <= 2:
    if k == 0:
        lb = color_bgr[0,0,k] - 10
        k += 1
    elif k > 0:
        lb = color_bgr[0,0,k] - 155
        k += 1
    lower_bound.append(lb)
k = 0
while k <= 2:
    if k == 0:
        ub = color_bgr[0,0,k] + 10
        k += 1
    elif k > 0:
        ub = color_bgr[0,0,k]
        k += 1

    upper_bound.append(ub)

lowerBound = np.array(lower_bound)
upperBound = np.array(upper_bound)

# mask image
maskImage = cv.inRange(hsv, lowerBound, upperBound)

# combined image
combImage = cv.bitwise_and(img,img, mask= maskImage)

# display image using imshow
cv.imshow("Original Image", img)
cv.imshow("Mask Image", maskImage)
cv.imshow('Resulting Image', combImage)
k = cv.waitKey(0)   # define waitkey with press key "k"

