'''
Create face embeddings for all the faces in the dataset/train directory
'''

import pickle
from imutils import paths
import cv2
import face_recognition
import os
from datetime import date
from parameters import DLIB_FACE_ENCODING_PATH, \
                       DATASET_PATH, \
                       NUMBER_OF_JITTERS, \
                       FACE_EMBEDDING_MODEL
                       
new_filename = DLIB_FACE_ENCODING_PATH + date.today().strftime('%m-%d-%Y') + ".pkl"
folder = "/dataset"

def create_face_embeddings():
    '''
    This function creates face encodings for all the faces in the dataset/train directory
    '''
    imagePaths = list(paths.list_images(DATASET_PATH))
    print(imagePaths)

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        print(name)
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)

        encoding = face_recognition.face_encodings(image, 
                                                   num_jitters=NUMBER_OF_JITTERS, # Higher number of jitters increases the accuracy of the encoding
                                                   model=FACE_EMBEDDING_MODEL)[0] #model='large' or 'small'
        knownEncodings.append(encoding)
        knownNames.append(name)
         
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    # create embeddings with today date included in filename
    f = open(new_filename, "wb")
    f.write(pickle.dumps(data))
    f.close()
    merge_face_embeddings()

def merge_face_embeddings():
    folder = "dataset/"
    with open(new_filename, "rb") as new:
        d1 = pickle.load(new)
        
    with open(folder + "dlib_face_encoding_10_07_2023.pkl", "rb") as old:
        d2 = pickle.load(old)
    
    merged = {}
    for key in d1:
        merged[key] = d1[key] + d2[key]
    
    with open(folder+"dlib_face_encoding.pkl", "wb") as merged_embeddings:
        pickle.dump(merged, merged_embeddings)
        
    
"""def rename_keys():
    
    with open("new_face_embeddings.pkl","rb") as file:
        d1 = pickle.load(file)
    
    d2 = {}
    
    for key, value in d1.items():
        if key == new_filename:
            d2["encodings"] = value
        else:
            d2["names"] = value
    with open(folder + "new_face_embeddings.pkl","wb") as file:
        pickle.dump(d2, file)"""


if __name__ == '__main__':
    create_face_embeddings()
