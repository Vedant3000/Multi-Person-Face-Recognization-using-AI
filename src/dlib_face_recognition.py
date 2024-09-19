# -*- coding: utf-8 -*-
"""
@Original Author: abhilash
@Modified by: Anubhav Patrick
"""

# importing the required libraries
import pickle
import os
import time
import cv2
import face_recognition
import numpy as np

from database_pandas import store_inferred_face_in_dataframe
from parameters import NUMBER_OF_JITTERS, \
    NUMBER_OF_TIMES_TO_UPSAMPLE, \
    DLIB_FACE_ENCODING_PATH, \
    FRAME_HEIGHT, FRAME_WIDTH, \
    FACE_MATCHING_TOLERANCE, \
    BATCH_SIZE, \
    INFERENCE_BUFFER_SIZE, \
    FACE_EMBEDDING_MODEL, \
    FRAME_DOWNSAMPLE, \
    NUMBER_OF_TIMES_TO_UPSAMPLE, \
    FACE_DETECTION_MODEL, \
    FACE_MATCHING_TOLERANCE, \
    BOUNDING_BOX_AREA_THRESHOLD
from custom_logging import logger
from locks import lock
from util.generic_utilities import get_bounding_box_size, \
    roi_face_detection, \
    create_mask

# load the known faces and embeddings
logger.info("[INFO] loading encodings...")
data = pickle.loads(open("dataset/dlib_face_encoding.pkl", "rb").read())

# save the encodings and the corresponding labels in seperate arrays in the same order
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# initialize the array variable to hold all face locations, encodings and names
all_face_locations = []
all_face_encodings = []
all_face_names = []
all_processed_frames = []
fp_count = 0

def single_frame_face_recognition(frame):
    '''Single frame face recognition function

    Arguments:
        frame {numpy array} -- frame to be processed

    Returns:
        a processed frame
    '''

    global fp_count

    if FRAME_DOWNSAMPLE:
        # resize the frame to FRAME_WIDTH*FRAME_HEIGHT to display the video if frame is too big
        # frame.shape[1] is width and frame.shape[0] is height
        print("Frame size before resize:",frame.shape)
        if frame.shape[1] > FRAME_WIDTH or frame.shape[0] > FRAME_HEIGHT:
            current_frame_small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        else:
            current_frame_small = frame
        print("Frame size after resize:",current_frame_small.shape)
    else:
        # consider the frame as it is
        current_frame_small = frame

    # For testing
    roi_left, roi_top, roi_right, roi_bottom = roi_face_detection(140, 200, current_frame_small.shape[1], current_frame_small.shape[0], 0, 0)
    mask = create_mask(current_frame_small, roi_left, roi_top, roi_right, roi_bottom)
    # Apply mask to the frame
    current_frame_small_mask = current_frame_small * mask

    # detect all faces in the image
    # arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(
        current_frame_small_mask, NUMBER_OF_TIMES_TO_UPSAMPLE, FACE_DETECTION_MODEL)

    # detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small_mask,
                                                         all_face_locations,
                                                         num_jitters=NUMBER_OF_JITTERS,
                                                         model=FACE_EMBEDDING_MODEL)
    # looping through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):

        # splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location

        # also get the bounding box size
        bounding_box_size = get_bounding_box_size(top_pos, right_pos, bottom_pos, left_pos)

        # find all the matches and get the list of matches
        all_matches = face_recognition.face_distance(known_face_encodings, current_face_encoding)
        # Find the best match (the smallest distance to a known face)
        best_match_index = np.argmin(all_matches)

        # check if face is close enough to match a known face
        if bounding_box_size >= BOUNDING_BOX_AREA_THRESHOLD:
            # If the best match is within tolerance, use the name of the known face
            if all_matches[best_match_index] <= FACE_MATCHING_TOLERANCE:
                name_of_person = known_face_names[best_match_index]
                # save the frame in which the name of person matches with Alok_2002901550002 or Kriti_2005110100056
                if name_of_person == "Alok_2002901550002" or name_of_person == "Kriti_2005110100056":
                    fp_count = fp_count + 1
                    OUTPUT_FILENAME = f"FP_{fp_count}_{name_of_person}.jpg"
                    cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 0), 2)
                    cv2.putText(current_frame_small, f'Name: {name_of_person}', (left_pos, bottom_pos), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                    cv2.imwrite(OUTPUT_FILENAME, current_frame_small)

                # save the name of the person in the dataframe
                store_inferred_face_in_dataframe(name_of_person, all_matches[best_match_index])
                color = (0, 255, 0)  # Green color for known face
            else:
                name_of_person = 'Unknown'
                color = (0, 0, 255)  # Red color for unknown face
        else:
            name_of_person = 'Too Far'
            color = (0, 165, 255)  # Orange color for far away face

        # draw rectangle around the face
        cv2.rectangle(current_frame_small, (left_pos, top_pos), (right_pos, bottom_pos), color, 5)

        # display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame_small, f'{name_of_person}, BBox:{bounding_box_size}', (
            left_pos, bottom_pos), font, 0.5, (255, 255, 255), 1)

    # draw rectangle around the roi
    roi_bb_color = (255, 0, 0)  # Blue color for roi
    cv2.rectangle(current_frame_small, (roi_left, roi_top),
                  (roi_right, roi_bottom), roi_bb_color, 5)

    yield current_frame_small


def multi_frame_face_recognition(frames_buffer):
    '''
        Multi frame face recognition function

        Arguments:
            frames_buffer {list} -- list of frames to be processed

        Yields:
            processed frames
    '''
    # pop first frame from frames_buffer to get the first frame
    while True:
        if len(frames_buffer) > 0:
            _ = frames_buffer.pop(0)
            break

    # Continue looping until there are no more frames to process.
    while True:

        # if is_stream:
        # check if there are frames in the buffer

        if len(frames_buffer) > 0:
          # pop first frame from frames_buffer
            img0 = frames_buffer.pop(0)
            if img0 is None:
                continue
            ret = True  # we have successfully read one frame from stream
            if len(frames_buffer) >= INFERENCE_BUFFER_SIZE:
                # clear the buffer if it has more than INFERENCE_BUFFER_SIZE frames to avoid memory overflow
                frames_buffer.clear()
        else:
            # buffer is empty, nothing to do
            continue

        # if we are able to read a frame then process it
        if ret:
            # yield the processed frame to the main thread
            yield from single_frame_face_recognition(frame=img0)


def batched_frame_face_recognition(frames_buffer):
    '''This function is to be used for GPU based face recognition. It performs face detection on a batch of frames at a time.

    Arguments:
        frames_buffer {list} -- list of frames to be processed

    Returns:
        None
    '''
    while True:

        # Wait until there are at least BATCH_SIZE frames in frames_buffer
        while len(frames_buffer) < BATCH_SIZE:
            time.sleep(0.01)

        # Find start time for batch processing
        tick = time.time()

        # Slice first BATCH frames from frames_buffer
        with lock:
            batched_frame_buffer = frames_buffer[:BATCH_SIZE]
            # Remove first BATCH_SIZE frames from frames_buffer
            del frames_buffer[:BATCH_SIZE]

        # extract batch of frames, cam names and cam ips from the batched_frame_buffer
        batch_of_frames = [batch[0] for batch in batched_frame_buffer]
        batch_of_cam_names = [batch[1] for batch in batched_frame_buffer]
        batch_of_cam_ips = [batch[2] for batch in batched_frame_buffer]

        # Use exception handling to catch any errors that might occur
        try:
            batch_of_face_locations = face_recognition.batch_face_locations(batch_of_frames,
                                                                            number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                                            batch_size=BATCH_SIZE)

            for i, all_face_locations_single_frame in enumerate(batch_of_face_locations):

                # detect face encodings for all the faces detected
                all_face_encodings_single_frame = face_recognition.face_encodings(batch_of_frames[i],
                                                                                  all_face_locations_single_frame,
                                                                                  num_jitters=NUMBER_OF_JITTERS,
                                                                                  model=FACE_EMBEDDING_MODEL)

                # looping through the face locations and the face embeddings
                for single_face_location_single_frame, current_face_encoding in zip(all_face_locations_single_frame, all_face_encodings_single_frame):

                    # print(f'Single face location: {single_face_location_single_frame}')
                    # print(f'Single face bounding box size: {get_bounding_box_size(*single_face_location_single_frame)}')
                    # To do - Put a threshold on the bounding box size to filter out small faces

                    # find all the matches and get the list of matches
                    all_matches = face_recognition.face_distance(
                        known_face_encodings, current_face_encoding)
                    # Find the best match (smallest distance to a known face)
                    best_match_index = np.argmin(all_matches)
                    # If the best match is within tolerance, use the name of the known face
                    if all_matches[best_match_index] <= FACE_MATCHING_TOLERANCE:
                        name_of_person = known_face_names[best_match_index]
                        # save the person details in the dataframe
                        store_inferred_face_in_dataframe(
                            name_of_person, all_matches[best_match_index], batch_of_cam_names[i], batch_of_cam_ips[i])

        except Exception as e:
            logger.error('Error in batch_face_locations: {}'.format(e))

        # Find end time for batch processing
        tock = time.time()

        # Calculate time taken for batch processing
        time_taken = tock - tick
        logger.info(
            f'Time taken for batch processing of {BATCH_SIZE} frames = {time_taken} seconds')

        # Todo: Purge frames_buffer if it has more than INFERENCE_BUFFER_SIZE frames to avoid memory overflow
        frames_buffer_size = len(frames_buffer)
        if frames_buffer_size > INFERENCE_BUFFER_SIZE:
            with lock:
                logger.warning(
                    f'Frames buffer size: {frames_buffer_size}. Purging frames_buffer...')
                frames_buffer.clear()
