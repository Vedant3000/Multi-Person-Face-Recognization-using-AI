'''This module writes the data to a file'''

import os
import time

import numpy as np


def write_to_file(file_name, data: dict, cnt: int):
    '''This function writes the all_faces_recognized to a file'''
    # get current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    high_probability = ''
    low_probability = ''
    # sort data based on dict values
    data = dict(
        sorted(data.items(), key=lambda item: item[1][0], reverse=True))

    # Remove wrongly classified faces older than 30 frames
    for key in list(data.keys()):
        # if the count of the face is less than equal to 3
        if data[key][0] <= 3:
            # delete items whose cnt - value[1] is greater than 30
            # that is those faces whose count is upto 3 but were detected earlier than 30 frames
            if cnt - data[key][1] > 30:
                del data[key]

    with open(file_name, 'w') as f:
        str = f'-----\nAttendance At: {current_time}\n\n-----\n'
        for key, value in data.items():
            if value[0] > 3:
                high_probability += f'{key} - {value[0]}\n'
            else:
                low_probability += f'{key} - {value[0]}\n'
        str += f'High Probability:\n{high_probability}\n\nLow Probability (Can be Ignored):\n{low_probability}\n------'
        f.write(f'{str}\n')


def check_for_directory(directory_path: str):
    '''This function checks if the directory exists, if not it creates the directory

    Args:
        directory_path (str): The path of the directory to be checked

    Returns:
        bool: True if the directory is created, False if the directory already exists
    '''

    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return True
    return False


def get_bounding_box_size(top, right, bottom, left):
    '''This function returns the size of the bounding box

    Args:
        top (int): top coordinate of the bounding box
        right (int): right coordinate of the bounding box
        bottom (int): bottom coordinate of the bounding box
        left (int): left coordinate of the bounding box

    Returns:
        int: size of the bounding box
    '''
    return (right - left) * (bottom - top)


def roi_face_detection(roi_width: int, roi_height: int, frame_width: int, frame_height: int, x: int = 0, y: int = 0):
    ''' This function returns the coordinates of the region of interest for face detection. 
        Face detection will be performed on the region of interest only.

    Args:
        roi_width (int): width of the region of interest
        roi_height (int): height of the region of interest
        frame_width (int): width of the frame
        frame_height (int): height of the frame
        x (int): x coordinate of the middle of the region of interest
        y (int): y coordinate of the bottom of the region of interest

    Returns:
        tuple(roi_left, roi_top, roi_right, roi_bottom): coordinates of the region of interest
    '''

    frame_x_center = frame_width // 2
    roi_width_half = roi_width // 2

    roi_left = frame_x_center - roi_width_half + x
    roi_right = frame_x_center + roi_width_half + x

    roi_bottom = frame_height - y
    roi_top = roi_bottom - (roi_height + y)

    return (roi_left, roi_top, roi_right, roi_bottom)


def create_mask(frame: np.ndarray, left: int, top: int, right: int, bottom: int):
    '''This function creates a mask for the frame so that face detection is performed only on the region of interest

    Args:
        frame (numpy.ndarray): frame on which the mask will be applied
        left (int): left coordinate of the region of interest
        top (int): top coordinate of the region of interest
        right (int): right coordinate of the region of interest
        bottom (int): bottom coordinate of the region of interest

    Returns:
        numpy.ndarray: mask for the frame
    '''
    mask = np.zeros_like(frame)
    mask[top:bottom, left:right] = 1
    return mask
