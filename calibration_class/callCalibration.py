## Call CameraCalibration Class Test Code ##

#%%
import sys
import numpy as np
import cv2 as cv
import glob
from matplotlib import pyplot as plt

sys.path.insert(1, 'calibration_class')
# import cameraCalibration

from cameraCalibration import CameraCalibration

# set calibration configuration
calibrationType = "Auto"
imgDimension = (7,7)
imgDir = "C:/Users/PTHAWAI/OneDrive - Daimler Truck/Documents/LUCIDVisionWorkspace/images"
instance1 = CameraCalibration(calibrationType=calibrationType, imgDimension=imgDimension, imgDir=imgDir)
#instance2 = ColorCalibration(green_hsv)

#%%
### Undistort Image ###

images = glob.glob(instance1.imgPath)
for image in images:

    img = cv.imread(image)
    h,w = img.shape[:2]
    mtxNew,roi = cv.getOptimalNewCameraMatrix(instance1.cameraMatrix, instance1.distCoeffs, (w,h), 1, (w,h))

    # using undistort
    imgUndist = cv.undistort(img, instance1.cameraMatrix, instance1.distCoeffs, None, mtxNew)
    x,y,w,h = roi
    imgUndist = imgUndist[y:y+h, x:x+w]


    plt.subplot(1,2,1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2), plt.imshow(imgUndist)
    plt.title('Undistorted'), plt.xticks([]), plt.yticks([])
    plt.show()

#%%
## Pose Estimatation ##

# create draw function to generate 3D axis
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(img, corner, tuple(imgpts[0].ravel().astype(int)), (255,0,0), 11)
    img = cv.line(img, corner, tuple(imgpts[1].ravel().astype(int)), (0,255,0), 11)
    img = cv.line(img, corner, tuple(imgpts[2].ravel().astype(int)), (0,0,255), 11)

    return img

# draw cube function
def drawCube(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 11)

    # draw top layer
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),11)

    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((imgDimension[0]*imgDimension[1],3), np.float32)
objp[:,:2] = np.mgrid[0:imgDimension[0],0:imgDimension[1]].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisCube = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

for image in glob.glob(instance1.imgPath):

    img = cv.imread(image)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, imgDimension ,None)

    if ret == True:
        
        # refine corners
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, instance1.cameraMatrix, instance1.distCoeffs)

        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, instance1.cameraMatrix, instance1.distCoeffs)
        imgNorm = draw(img, corners2, imgpts)
        plt.subplot(1,2,1), plt.imshow(imgNorm)
        plt.title('Pose Estimate'), plt.xticks([]), plt.yticks([])

        imgptsCube, jacCube = cv.projectPoints(axisCube, rvecs, tvecs, instance1.cameraMatrix, instance1.distCoeffs)
        imgCube = drawCube(img, corners2, imgptsCube)
        plt.subplot(1,2,2), plt.imshow(imgCube)
        plt.title('Cubic Pose Estimate'), plt.xticks([]), plt.yticks([])
        plt.show()

    elif ret == False:
        print(f"Error: Could not detect corners of image {image}")

