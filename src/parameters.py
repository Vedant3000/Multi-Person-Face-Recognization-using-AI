'''
This module contain the default parameters used by the app 
'''

#The default sender of emails
EMAIL_SENDER = 'support.ai@giindia.com'

#The default receiver of emails
EMAIL_RECEIVER = 'anubhav.patrick@giindia.com'

# The default email subject
EMAIL_SUBJECT = 'Face Recognition Summary Report'

# The default email body
EMAIL_BODY = 'Please find the attached Face Recognition Summary Report'

#Path where dataset is stored (used for creating face embedding)
DATASET_PATH = 'dataset/train/new_dataset'

# The path where dlib face encodings are stored
DLIB_FACE_ENCODING_PATH = 'dataset/dlib_face_encoding_550_data.pkl'

# The path where face recognition report will be stored
REPORT_PATH = 'reports/inferred_faces.csv'

# Fles path where various supplementary files are stored
FILES_PATH = 'data/files/'

# The path of the demo video
VIDEO_PATH = 'data/files/vid.mp4'

# The path where the user video will be uploaded
VIDEO_UPLOAD_PATH = 'src/static/video/vid.mp4'

# The path where logs will be stored
LOG_FILE_PATH = 'logs/multicam_server.log'

# set whether or not to downsampling the frame (for faster processing)
FRAME_DOWNSAMPLE = True

# Set video frame height and width
FRAME_HEIGHT = 360 #720 #576
FRAME_WIDTH = 640 #1280 #1024

# set BATCH_SIZE for face detection
BATCH_SIZE = 1 #for DGX 32, 1 for CPU

# buffer size for video streaming to minimize inconsistent network conditions
LIVE_STREAM_BUFFER_SIZE = 2560 #single camera

# buffer size for frames on which face recognition will be performed
INFERENCE_BUFFER_SIZE = 2560 #8 for DGX

# IP Camera Details
IP_CAMS = {#'cam1': 'http://192.168.12.10:4747/video',
           'cam2': 'http://192.168.19.164:4747/video',
          }

# Set wait duration for IP cam re initialization if we are not able to initialize the cam
IP_CAM_REINIT_WAIT_DURATION = 30 #seconds

# The bounding box area threshold for face detection below which the face will be ignored
BOUNDING_BOX_AREA_THRESHOLD = 1500

# Time of the day when the email will be sent
EMAIL_SEND_TIME = '18:00' # 6 PM

# Wait duration for the email report to be sent
EMAIL_SEND_WAIT_DURATION = 15 #seconds

##############################################################################
# Parameters specific to dlib face recognition
##############################################################################
# face matching tolerance (distance -> less the distance, more the similarity)
FACE_MATCHING_TOLERANCE = 0.45

# Range (Max - Min) Tolerance to reduce false positives
RANGE_TOLERANCE = 0.036

#face recognition model
FACE_DETECTION_MODEL = 'cnn' #hog -> for CPU or cnn -> for GPU (DGX)

# Number of times to upsample the image looking for faces
NUMBER_OF_TIMES_TO_UPSAMPLE = 1 # for realtime keep it to 1

# Face embedding model
FACE_EMBEDDING_MODEL = 'large'

# Number of jitters (random shifts) to use when creating the face encoding
NUMBER_OF_JITTERS = 1 #25
##############################################################################
