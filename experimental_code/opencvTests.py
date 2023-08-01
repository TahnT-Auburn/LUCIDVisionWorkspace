## OpenCV Testing Environment ##


## OPENCV-PYTHON TUTORIALS ##
# source: https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html

#%%
import cv2 as cv
import sys
import numpy as np
from matplotlib import pyplot as plt

#%%
## GUI FEATURES: Getting Started with Images ##
#
# Goals:
# 1) Read an image from file using imread
# 2) Display an image in an OpenCV window using imshow
# 3) Write an image to a file using imwrite

# define an image using imread
img = cv.imread(cv.samples.findFile("images/kakashi.jpg"))

# check if image is loaded correctly
if img is None:
    sys.exit("Could not read the image")

# display image using imshow
# cv.imshow("Original Image", img)
# k = cv.waitKey(0)   # define waitkey with press key "k"

'''
# write image using imwrite
# NOTE: if the the pressed key "k" is "s" then write image
if k == ord("s"): 
    cv.imwrite("images/image_0.png", img)
'''

#%%
## IMAGE PROCESSING: Changing Colorspaces
#
# Goals:
# 1) covert images from one color-space to another (BGR -> Gray), (BGR -> HSV), etc.
# 2) extract a colored object

# display available flags (flags determines the color conversion type)
dispFlag = 'true'
if dispFlag == 'true':
    flags = [i for i in dir(cv) if i.startswith('COLOR_')]
    print(flags)
elif dispFlag == 'false':
    pass
else:
    print('Not valid entry for dispFlag')

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
color_bgr = hsv_blue

# convert BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# generate HSV bounds
lower_bound = []
upper_bound = []

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

#%%
## IMAGE PROCESSING: Image Thresholding
#
# Goals:
# 1) Learn simple thresholding, adaptive thresholding, and Otsu's thresholding

# ----- Simple Thresholding -----
img_gray = cv.imread('images/shura.jpg', cv.IMREAD_GRAYSCALE)
assert img_gray is not None, "file could not be read, check with os.path.exists()"

ret,thresh1 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img_gray,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img_gray,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img_gray,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# ----- Adaptive thresholding -----
adpt_thresh1 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
adpt_thresh2 = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

titles = ['Original', 'Global', 'Adaptive Mean', 'Adaptive Gaussian']
images = [img_gray, thresh1, adpt_thresh1, adpt_thresh2]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# ----- Otsu's Thresholding -----
ret2, otsu_thresh1 = cv.threshold(img_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img_gray,(5,5),0)
ret3, otsu_thresh2 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img_gray, 0, thresh1, img_gray, 0, otsu_thresh1, blur, 0, otsu_thresh2]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
 'Original Noisy Image','',"Otsu's Thresholding",
 'Gaussian filtered Image','',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

#%%
# IMAGE PROCESSING: Image Gradients

# load image
img = cv.imread(cv.samples.findFile("images/image_0.jpg"), cv.IMREAD_GRAYSCALE)

# three gradient filter provided by OpenCV
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()

# compensating for white-to-black (negative slope) transitions

# Output dtype = cv.CV_8U
sobelx8u = cv.Sobel(img,cv.CV_8U,1,0,ksize=5)

# Output dtype = cv.CV_64F. Then take its absolute and convert to cv.CV_8U
sobelx64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()

# %% IMAGE PROCESSING: Canny Edge Detection

img = cv.imread('images/checkerboard.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# canny edge detection
edges = cv.Canny(img,100,200)

# dilate edge image
kernel = np.ones((5,5),np.uint8)    # define kernel
edges_dilate = cv.dilate(edges, kernel, iterations=1)
plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(edges_dilate,cmap = 'gray')
plt.title('Dilated Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# %%
