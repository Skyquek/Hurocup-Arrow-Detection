# USAGE
# python generate_dataset.py --input videos/left.mp4 --output dataset/left --skip 2
# python generate_dataset.py --input videos/right.mp4 --output dataset/right --skip 2
# python generate_dataset.py --input videos/up.mp4 --output dataset/up --skip 2

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
from preprocessing import pre_processing

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to input videos")
ap.add_argument("-o", "--output", type=str, required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# open a pointer to the videos file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# loop over frames from the videos file stream
while True:
    # grab the frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # increment the total number of frames read thus far
    read += 1

    # check to see if we should process this frame
    if read % args["skip"] != 0:
        continue

    # write the frame to disk
    p = os.path.sep.join([args["output"], "{}.png".format(saved)])

    # rotate frame to the right
    # optional
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    contour_details = pre_processing(frame)
    if len(contour_details['bounding_box']) != 0:
        x = contour_details['bounding_box'][0][0]
        y = contour_details['bounding_box'][0][1]
        w = contour_details['bounding_box'][0][2]
        h = contour_details['bounding_box'][0][3]
        ROI = frame[y:y + h, x:x + w]
        frame = ROI

        cv2.imwrite(p, frame)
        saved += 1
        print("[INFO] saved {} to disk".format(p))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()
