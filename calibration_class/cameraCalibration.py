'''
#------------------------------------------------------------------------------------------------------
### LUCID VISION CAMERA CALIBRATION ###

Author: Tahn Thawainin

Description:  CameraCalibration class uses LUCID Vision's Arena API and OpenCV to 
              calibrate a LUCID Vision camera. This class allows the user
              to calibrate using preset images ("Auto") or capturing a new set of images
              ("Manual"). CameraCalibration returns the camera matrix, distortion coefficients,
              rotation and translation vectors. RMS and Re-projection errors are also calculated.

Resources:   https://thinklucid.com/ (LUCID Vision website)  
             https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html (Online OpenCV documentation)

#--------------------------------------------------------------------------------------------------------
'''

from arena_api.system import system # Arena API  
from arena_api.buffer import *
from arena_api.enums import PixelFormat

from matplotlib import pyplot as plt

import ctypes
import numpy as np
import cv2 as cv
import time
import glob


class CameraCalibration:
    
    def __init__(self, calibrationType:str="Manual", imgDimension:tuple=None, imgDir:str=None):
        
        assert calibrationType == "Auto" or calibrationType == "Manual",\
        f"First input calibrationType {calibrationType} is not a valid input. \n Expected calibrationType inputs: \n \"Auto\", \"Manual\""
        assert imgDimension[0] > 0 and imgDimension[1] > 0,\
        f"Third input imgDimensions {imgDimension} is not a valid input. \n Both elements of imgDimensions must be greater than 0"
        
        # set attributes
        self.calibrationType = calibrationType
        self.imgDimension = imgDimension
        self.imgDir = imgDir

        if self.calibrationType == "Auto":
            
            print("Auto calibration Started...")

            # execute actions
            CameraCalibration.calibrateSetup(self)
            CameraCalibration.calibrate(self)
            CameraCalibration.reprojectionError(self)

            print("Auto calibration ended successfully")

        elif self.calibrationType == "Manual":
            
            print("Manual calibration Started...")

            # execute methods
            CameraCalibration.createDevice(self)
            CameraCalibration.resetNodes(self)
            CameraCalibration.setupDevice(self)
            CameraCalibration.startStream(self)
            CameraCalibration.calibrateSetup(self)
            CameraCalibration.calibrate(self)
            CameraCalibration.reprojectionError(self)

            print("Manual calibration ended successfully")

    def calibrateSetup(self):
        '''
        Method calibrateSetup takes in images of a checkerboard
        from a specified folder and runs them through a calibration setup.
        NOTE: At least 10 images should satisfy the setup criteria
        '''
        
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points
        objp = np.zeros((self.imgDimension[0]*self.imgDimension[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:self.imgDimension[0],0:self.imgDimension[1]].T.reshape(-1,2)

        # define arrays to store object and image points from successful images
        self.objPoints =[]
        self.imgPoints = []

        # if calibratioType set to "Auto", specify path to images by updating self.imgPath
        # NOTE: if calibrationType set to "Manual", self.imgPath is set in method startStream
        if self.calibrationType == "Auto":
            imgFolder = "/calibration_images"
            imgExtension = "/*.jpg"
            self.imgPath = self.imgDir + imgFolder + imgExtension

        images = glob.glob(self.imgPath)
        succCount = 0
        failCount = 0
        for image in images:

            # read image
            img = cv.imread(image)
            self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # find chessboard corners
            ret, corners = cv.findChessboardCorners(self.gray, self.imgDimension, None)

            # if corner detection is successful, populate object and image points
            if ret == True:

                self.objPoints.append(objp) # append object points
                cornersRefined = cv.cornerSubPix(self.gray, corners, (11,11), (-1,-1), criteria) # refine corners
                self.imgPoints.append(cornersRefined)   # append image points
                succCount += 1
            
            # display error message if corner detection fails
            else:
                
                print(f"Warning: {image} failed setup! Could not detect corners OR incorrect dimensions were given")
                failCount += 1

        print("Calibration Setup Complete")
        print(f"{succCount} out of {len(images)} image(s) succeeded")
        print(f"{failCount} out of {len(images)} image(s) failed")

        cv.destroyAllWindows()
    
    def calibrate(self):
        '''
        Method calibrate uses OpenCV's calibrateCamera to generate a camera matarix,
        distrotion coefficients, rotation and translation vectors.
        The function calibrateCamera uses the objectPoints and imagePoints generated from
        method calibrationSetup.
        NOTE: Higher quality, number, and distinct orientation of calibration images improves parameter estimates
        '''

        # call calibrateCamera function
        self.ret, self.cameraMatrix, self.distCoeffs, self.rotationVect, self.translationVect = cv.calibrateCamera(self.objPoints, self.imgPoints, self.gray.shape[::-1], None, None)
        
        # display calibration outputs
        print(f"Projection Error: \n{self.ret}\n")
        print(f"Camera Matrix: \n{self.cameraMatrix}\n")
        print(f"Distortion coefficients: \n{self.distCoeffs}\n")
        print(f"Camera Matrix: \n{self.cameraMatrix}\n")
        print(f"Rotation Vector: \n{self.rotationVect}\n")
        print(f"Translation Vector: \n{self.translationVect}\n")

    def reprojectionError(self):
        '''
        Method reprojectionError is used as a metric on calibration performance.
        It uses calibration outputs to reproject a set of image points and compares
        them to the image points generated in method calibrationSetup.
        '''

        self.mean_error = 0
        for i in range(len(self.objPoints)):
            imgPoints2, _ = cv.projectPoints(self.objPoints[i], self.rotationVect[i], self.translationVect[i], self.cameraMatrix, self.distCoeffs)
            error = cv.norm(self.imgPoints[i], imgPoints2, cv.NORM_L2)/len(imgPoints2)
            self.mean_error += error
        print("Re-projection Error: {}".format(self.mean_error/len(self.objPoints)))

    ### Manual Image Capture Methods ### 
    def createDevice(self):
        '''
        Method createDevice detects and creates a device.
        If no device is connected, createDevice waits for user
        to connect device before raising an exception.
        '''
        
        tries = 0
        tries_max = 6
        sleep_time_secs = 10
        while tries < tries_max:  # Wait for device for 60 seconds
            self.devices = system.create_device()
            if not self.devices:
                print(
                    f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                    f'secs for a device to be connected!')
                for sec_count in range(sleep_time_secs):
                    time.sleep(1)
                    print(f'{sec_count + 1 } seconds passed ',
                        '.' * sec_count, end='\r')
                tries += 1
            else:
                print(f'Created {len(self.devices)} device(s)')
                self.device = self.devices[0] # select device
                return self.devices, self.device
        else:
            raise Exception(f'No device found! Please connect a device and run '
                            f'the example again.')
        
    def resetNodes(self):
        '''
        Method resetNodes allows the user to reset nodes before setupDevice is called.
        Note that some nodes are dependent on others and not resetting them can cause
        issues when changing configurations for future streams.
        '''

        # define node map
        nodemap = self.device.nodemap

        # specify nodes to reset
        nodes = nodemap.get_node(['Width','Height','OffsetX','OffsetY'])

        # reset nodes
        nodes['Width'].value = nodes['Width'].max
        nodes['Height'].value = nodes['Height'].max
        nodes['OffsetX'].value = 0
        nodes['OffsetY'].value = 0 

    def setupDevice(self):
        '''
        Method setupDevice allows the user to configure camera nodes
        to change to setup parameters. See py_explore_nodes/nodemaps/nodetypes
        for more information on nodes
        '''

        nodemap = self.device.nodemap
        #nodemap['Width'].value = 1280 #nodemap['Width'].max
        #nodemap['Height'].value = 720 #nodemap['Height'].max
        nodemap['PixelFormat'].value = PixelFormat.BGR8
        nodemap['Gain'].value = 40.0

        # set channels (adjust based on PixelFormat)
        self.num_channels = 3

        """
        set width and height to max values might make the video frame rate low
        The larger the height of the buffer the lower the fps
        """
        
        width_node = nodemap['Width']
        width = nodemap['Width'].max // 3

        # get a value that aligned with node increments
        while width % width_node.inc:
            width -= 1
        nodemap['Width'].value = width

        height_node = nodemap['Height']
        height = nodemap['Height'].max // 3

        # get a value that aligned with node increments
        while height % height_node.inc:
            height -= 1
        nodemap['Height'].value = height
        
        # set camera offset
        # NOTE: Available options vary depending on the width and height of the image
        #       Check the minimum and maximum values if errors occur
        #       Adjust in increments of 4
        nodemap['OffsetX'].value = 700
        nodemap['OffsetY'].value = 0
        
        # For performance ---------------------------------------------------------
        
        # make sure the device sends images continuously
        nodemap['AcquisitionMode'].value = 'Continuous'

        # automate the calculation of max FPS whenever the device settings change
        nodemap['AcquisitionFrameRateEnable'].value = True

        # set FPS node to max FPS which was set to be automatically calculated
        # base on current device settings
        nodemap['AcquisitionFrameRate'].value = nodemap['AcquisitionFrameRate'].max

        # max FPS according to the current settings
        nodemap['DeviceStreamChannelPacketSize'].value = nodemap['DeviceStreamChannelPacketSize'].max
        
        # -------------------------------------------------------------------------

        # Stream nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap
        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True
    
    def startStream(self):
        '''
        Method startStream initiates streams and displays live streaming windows to allow user
        to capture and save images to a specified folder. The image path self.imgPath is automatically
        updated and returned at when user exits the streams.
        '''

        # define path to save images 
        imgFolder = "/calibration_images"
        imgExtension ="/*.jpg"
        self.imgPath = self.imgDir + imgFolder + imgExtension

        curr_frame_time = 0
        prev_frame_time = 0

        with self.device.start_stream():
        
            print("Started stream... \n press 's' to capture/save image \n press 'esc' to quit stream")
            saveCount = 0

            while True:

                # Used to display FPS on stream
                curr_frame_time = time.time()

                # get buffer
                buffer = self.device.get_buffer()
            
                # Copy buffer and requeue to avoid running out of buffers
                item = BufferFactory.copy(buffer)
                self.device.requeue_buffer(buffer)

                buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))

                # Buffer data as cpointers can be accessed using buffer.pbytes
                array = (ctypes.c_ubyte * self.num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
                
                # Create a reshaped NumPy array to display using OpenCV
                npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))

                # generate stream windows
                fps = str(1/(curr_frame_time - prev_frame_time))
                #cv.putText(npndarray, fps, (7, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (100, 255, 0), 1, cv.LINE_AA)
                cv.imshow('Raw', npndarray)

                # Destroy the copied item to prevent memory leaks
                BufferFactory.destroy(item)
                
                # update frame time
                prev_frame_time = curr_frame_time

                # save image if 's' key is pressed
                key = cv.waitKey(1)
                if key == ord("s"):
                    saveCount +=1
                    # specify self.imgPath
                    # NOTE: make sure cv.imwrite saves images to the same path as self.imgPath!
                    imgName = f"/calibration_image_{saveCount}.jpg"
                    cv.imwrite(self.imgDir + imgFolder + imgName, npndarray)
                    print(f"Saved {imgName}_{saveCount}.jpg to path {self.imgDir + imgFolder}")
                # break if 'esc' key is pressed
                elif key == 27:
                    print(f"{saveCount} image(s) saved to {self.imgDir + imgFolder}")
                    break

            # quit stream
            self.device.stop_stream()
            cv.destroyAllWindows()
            print("Ended stream")

        system.destroy_device()

        return self.imgPath