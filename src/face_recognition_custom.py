''' This module decouples face recognition logic from rest of the application'''
from parameters import FACE_MATCHING_TOLERANCE, NUMBER_OF_TIMES_TO_UPSAMPLE, BATCH_SIZE, FACE_DETECTION_MODEL, NUMBER_OF_JITTERS
import cv2
import numpy as np
import pickle
import csv
import os

def batched_face_detection(batch_of_frames : list):
    ''' This function takes a batch of frames and returns face locations for each frame in the batch
    
    Arguments:
        batch_of_frames {list} -- list of frames
        
    Returns:
        list -- list of face locations for each frame in the batch
    '''

    # Customize following lines of code to get face locations from a batch of frames
    
    # import required libraries
    import face_recognition

    # get the batch of face locations
    batch_of_face_locations = face_recognition.batch_face_locations(batch_of_frames,
                                                                   number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                                   batch_size=BATCH_SIZE)

    # return the batch of face locations
    return batch_of_face_locations


def face_detection(frame : list):
    ''' This function takes a frame and returns face locations for each face in the frame
    
    Arguments:
        frame {list} -- list of frames
        
    Returns:
        list -- list of face locations for each face in the frame
    '''

    # Customize following lines of code to get face locations from a frame

    # import required libraries
    import face_recognition

    # get the face locations
    face_locations_single_frame = face_recognition.face_locations(frame,
                                                        number_of_times_to_upsample=NUMBER_OF_TIMES_TO_UPSAMPLE,
                                                        model=FACE_DETECTION_MODEL)

    # return the face locations
    print('face location', face_locations_single_frame)
    return face_locations_single_frame


def face_embedding(frame, face_locations=None):
    ''' This function takes a frame containing one or more faces and returns face encodings for each face in the frame
    
    Arguments:
        frame {np.array} -- a frame
        
    Returns:
        list of {np.array} -- list of face encodings for each face in the frame
    '''

    # Customize following lines of code to get face encodings from a frame

    # import required libraries
    import face_recognition
    from parameters import FACE_EMBEDDING_MODEL, NUMBER_OF_JITTERS
    #print(type(frame))
    # get the face encodings
    #print(type(face_locations))
    face_encodings_single_frame = face_recognition.face_encodings(frame, 
                                                                  face_locations,
                                                                  NUMBER_OF_JITTERS
                                                                  )

    # return the face encodings of all faces in the frame
    #print('face encoding ', face_encodings_single_frame)
    return face_encodings_single_frame


def face_distance(frames_buffer, batch_of_face_locations, known_face_encodings, folder_name):
    ''' This function takes a list of face encodings and a single face encoding and returns the distance between each face encoding and the single face encoding
    
    Arguments:
        known_face_encodings {list of {np.array}} -- a list of face encodings
        current_face_encoding {np.array} -- a single face encoding
        
    Returns:
        list of {float} -- list of distances between each face encoding and the single face encoding
    '''

    # Customize following lines of code to get face distances from a list of face encodings and a single face encoding
    global existing_data, file_path
    # import required libraries
    import face_recognition
    inference_name_distance = [[folder_name, 0]]
    #csv_file_path = 'src/Inference/'
    #csv_file_path = f"{csv_file_path}/{filename}"
    for current_frame_small, all_face_locations in zip(frames_buffer,batch_of_face_locations):
        all_face_encodings = face_recognition.face_encodings(current_frame_small, 
                                                                    all_face_locations,
                                                                    NUMBER_OF_JITTERS
                                                                    )

        for current_face_location, current_face_encoding in zip(all_face_locations,all_face_encodings):
            # get the face distances
            #print(type(known_face_encodings) , type(current_face_encoding))
            face_distances = face_recognition.face_distance(known_face_encodings, current_face_encoding)
            
            '''distances = face_distances.copy()
            min_value = distances[0]
            for i in range(5):
                min_value = distances[0]
                for j in range(len(distances)):
                    if min_value > distances[j]:
                        min_value = distances[j]
                        distances[j] = 1
                        index = j
                #print('Name', known_face_names[index] , 'Face distance' , min_value)'''
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] <= FACE_MATCHING_TOLERANCE:
                name_of_person = known_face_names[best_match_index]
                inference_name_distance.append([known_face_names[best_match_index],face_distances[best_match_index]])
    print('inference ', inference_name_distance)
    #combined_data = existing_data + inference_name_distance
    if len(inference_name_distance) > 1:
        # Write the combined data back to the CSV in append mode
        with open(file_path, mode='a', newline='\n') as file:
            writer = csv.writer(file)
            writer.writerows(inference_name_distance)
        

folder_path = "output3/16/"
image_files = os.listdir(folder_path)
frames_buffer = []
for image_file in image_files:
    image_path = f"{folder_path}{image_file}"
    frame = cv2.imread(image_path)
    print(image_path)
    frames_buffer.append(frame)
batch_of_face_locations = batched_face_detection(frames_buffer)
print(len(batch_of_face_locations))
# Loading the embeddings
known_face_embeddings = 'dlib_face_encoding_550_data.pkl'
data = pickle.loads(open(known_face_embeddings,"rb").read())
#save the embeddings and the corresponding labels in seperate arrays in the same order
known_face_embeddings = data["encodings"]
known_face_names = data["names"]
#print('stored encodings' , len(known_face_embeddings))
#prob = face_distance(known_face_embeddings, face_embeddings_single_frame)
#image = path.split('/')[-1].replace(".png" , ' ')
#print('For' , image , 'distance is ' , prob)
# File path to the existing CSV
file_path = "reports/inf3.csv"

# Read the existing inference-results from the CSV
existing_data = []
with open(file_path, mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        existing_data.append(row)
#print('prev inference ',existing_data)
folder_name = folder_path.split('/')[1]
print(folder_name)
face_distance(frames_buffer, batch_of_face_locations, known_face_embeddings, folder_name)
