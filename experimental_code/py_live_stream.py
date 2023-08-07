# -----------------------------------------------------------------------------
# Copyright (c) 2022, Lucid Vision Labs, Inc.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# -----------------------------------------------------------------------------

#%%
from arena_api.system import system
from arena_api.buffer import *
from arena_api.enums import PixelFormat
from arena_api.__future__.save import Writer

import ctypes
import numpy as np
import cv2 as cv
import time

'''
Live Stream: Introduction
    This example introduces the basics of running a live stream 
    from a single device. This includes creating a device, selecting
    up stream dimensions, getting buffer cpointer data, creating an
    array of the data and reshaping it to fit image dimensions using
    NumPy and displaying using OpenCV-Python.
'''

#%%
def create_devices_with_tries():
    '''
    This function waits for the user to connect a device before raising
        an exception
    '''

    tries = 0
    tries_max = 6
    sleep_time_secs = 10
    while tries < tries_max:  # Wait for device for 60 seconds
        devices = system.create_device()
        if not devices:
            print(
                f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
                f'secs for a device to be connected!')
            for sec_count in range(sleep_time_secs):
                time.sleep(1)
                print(f'{sec_count + 1 } seconds passed ',
                    '.' * sec_count, end='\r')
            tries += 1
        else:
            print(f'Created {len(devices)} device(s)')
            return devices
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')
    
def resetNodes(device):
        '''
        Method resetNodes allows the user to reset nodes before setting them.
        Note that some nodes are dependent on others and not resetting them can cause
        issues when changing configurations for future streams
        '''

        # define node map
        nodemap = device.nodemap

        # specify nodes to reset
        nodes = nodemap.get_node(['Width','Height','OffsetX','OffsetY'])

        # reset nodes
        nodes['Width'].value = nodes['Width'].max
        nodes['Height'].value = nodes['Height'].max
        nodes['OffsetX'].value = 0
        nodes['OffsetY'].value = 0

#%%
def setup(device):
    """
    Setup stream dimensions and stream nodemap
        num_channels changes based on the PixelFormat
        Mono 8 would has 1 channel, RGB8 has 3 channels

    """
    # Set node values ---------------------------------------------------------

    nodemap = device.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'Gain'])

    #nodes['Width'].value = 1280
    #nodes['Height'].value = 720
    #nodes['PixelFormat'].value = 'RGB8's
    pixel_format = PixelFormat.BGR8
    nodes['PixelFormat'].value = pixel_format
    nodes['Gain'].value = 40.0

    # set channels (adjust based on PixelFormat)
    num_channels = 3

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
    device.nodemap['AcquisitionMode'].value = 'Continuous'

    # automate the calculation of max FPS whenever the device settings change
    nodemap['AcquisitionFrameRateEnable'].value = True

    """
    set FPS node to max FPS which was set to be automatically calculated
    base on current device settings
    """
    nodemap['AcquisitionFrameRate'].value = nodemap['AcquisitionFrameRate'].max

    """
    max FPS according to the current settings
    """
    nodemap['DeviceStreamChannelPacketSize'].value = nodemap['DeviceStreamChannelPacketSize'].max
    
    # -------------------------------------------------------------------------

    # Stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    return num_channels, pixel_format

#%%
def saveImage(buffer, pixel_format, saveCount):
    '''
    Method saveImage coverts an input buffer to a specificied pixel format and saves an image
    to a specified location in the directory. The saveNum variable refers to how many times
    the function has been executed to update the writer.pattern to be able to save multiple images.
    NOTE: Must update saveNum outside of the function
    '''

    # covert image to a displayable pixel format
    converted = BufferFactory.convert(buffer, pixel_format)
    print(f"Coverted image to {pixel_format}")

    # prepare image writer (requires 2 parameters to save an image: the buffer and the specified file name or pattern)
    writer = Writer()
    writer.pattern = f"images/testimages/image_{saveCount}"

    # save converted buffer
    writer.save(converted)
    print(f"{saveCount} image(s) saved to {writer.pattern}")

    # destroy converted buffer
    BufferFactory.destroy(converted)


#%%
def color_detection(npndarray, color_hsv):

    '''
    Color detection function using HSV coversions in OpenCV.
    Manually set BRG bounds to generate a mask image using cv.inRange

    Inputs: npndarray - the current frame that is displayed using OpenCV (Type: ndarray)
            color_hsv - the hsv array of the desired color for detection (Type: ndarray)
    '''

    # convert BGR to HSV
    hsv = cv.cvtColor(npndarray, cv.COLOR_BGR2HSV)

    # generate HSV bounds
    lower_bound = []
    upper_bound = []
    k = 0
    while k <= 2:
        if k == 0:
            lb = color_hsv[0,0,k] - 10
            k += 1
        elif k > 0:
            lb = color_hsv[0,0,k] - 155
            k += 1
        lower_bound.append(lb)
    k = 0
    while k <= 2:
        if k == 0:
            ub = color_hsv[0,0,k] + 10
            k += 1
        elif k > 0:
            ub = color_hsv[0,0,k]
            k += 1

        upper_bound.append(ub)

    lowerBound = np.array(lower_bound)
    upperBound = np.array(upper_bound)

    # generate mask image
    mask = cv.inRange(hsv, lowerBound, upperBound)

    # bitwise_and to combine mask and original image
    res = cv.bitwise_and(npndarray,npndarray, mask=mask)

    return mask, res

#%%
def edge_detection(npndarray):
    
    # canny edge detection
    edges = cv.Canny(npndarray,100,200)

    # dilate edge image
    kernel = np.ones((5,5),np.uint8)    # define kernel
    edges_dilate = cv.dilate(edges, kernel, iterations=1)

    return edges, edges_dilate

#%%
def example_entry_point():
    """
    demonstrates live stream
    (1) Start device stream
    (2) Get a buffer and create a copy
    (3) Requeue the buffer
    (4) Calculate bytes per pixel for reshaping
    (5) Create array from buffer cpointer data
    (6) Create a NumPy array with the image shape
    (7) Display the NumPy array using OpenCV
    (8) When Esc is pressed, stop stream and destroy OpenCV windows
    """

    devices = create_devices_with_tries()
    device = devices[0]

    # reset
    resetNodes(device)

    # Setup
    num_channels, pixel_format = setup(device)

    curr_frame_time = 0
    prev_frame_time = 0

    with device.start_stream():
        """
        Infinitely fetch and display buffer data until esc is pressed
        """
        saveCount = 0
        while True:
            # Used to display FPS on stream
            curr_frame_time = time.time()

            buffer = device.get_buffer()

            """
            Copy buffer and requeue to avoid running out of buffers
            """
            item = BufferFactory.copy(buffer)
            device.requeue_buffer(buffer)

            buffer_bytes_per_pixel = int(len(item.data)/(item.width * item.height))
            """
            Buffer data as cpointers can be accessed using buffer.pbytes
            """
            array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(ctypes.addressof(item.pbytes))
            """
            Create a reshaped NumPy array to display using OpenCV
            """
            npndarray = np.ndarray(buffer=array, dtype=np.uint8, shape=(item.height, item.width, buffer_bytes_per_pixel))

            # ----- color object detection -----

            # define BGR colors
            white = np.uint8([[[255, 255, 255]]])
            blue = np.uint8([[[255, 0, 0]]])
            green = np.uint8([[[0, 255, 0]]])
            red = np.uint8([[[0, 0, 255]]])
            
            # convert to HSV colors
            white_hsv = cv.cvtColor(white, cv.COLOR_BGR2HSV)
            blue_hsv = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
            green_hsv = cv.cvtColor(green, cv.COLOR_BGR2HSV)
            red_hsv = cv.cvtColor(red, cv.COLOR_BGR2HSV)

            # call color detection function
            mask, res = color_detection(npndarray, green_hsv)

            # call edge detection function
            edges, edges_dilated = edge_detection(npndarray)
            # -----------------------------------
            # stream
            fps = str(1/(curr_frame_time - prev_frame_time))
            cv.putText(npndarray, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('Original', npndarray)

            cv.putText(mask, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('Masked', mask)

            cv.putText(res, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('Color Isolated', res)

            cv.putText(edges, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('Original Edge Detection', edges)

            cv.putText(edges_dilated, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
            cv.imshow('Dilated Edge Detection', edges_dilated)

            """
            Destroy the copied item to prevent memory leaks
            """
            BufferFactory.destroy(item)

            prev_frame_time = curr_frame_time

            """
            Break if esc key is pressed
            """
            key = cv.waitKey(1)
            if key == 27:
                break
            elif key == ord("s"):
                #saveImage(buffer, pixel_format, saveCount)
                cv.imwrite(f"images/testimages/image_{saveCount+1}.jpg", npndarray)
                saveCount += 1
                
        device.stop_stream()
        cv.destroyAllWindows()
    
    system.destroy_device()

#%%
if __name__ == '__main__':
    print('\nWARNING:\nTHIS EXAMPLE MIGHT CHANGE THE DEVICE(S) SETTINGS!')
    print('\nExample started\n')
    example_entry_point()
    print('\nExample finished successfully')


