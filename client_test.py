import socket
import sys
import numpy
import struct
import binascii

from PIL import Image
from detector import Detector
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# parser.add_argument('-c', '--checkpoint',
#                     help=('directory to load checkpoint'))
# parser.add_argument('--instance-threshold', default=0.0, type=float,
#                     help='Defines the threshold of the detection score')
parser.add_argument('-d', '--downscale', default=8, type=int,
                    help=('downscale of the received image'))
# parser.add_argument('--square-edge', default=401, type=int,
#                     help='square edge of input images')

args = parser.parse_args()

# image data
downscale = args.downscale
width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels


# Set up detector
detector = Detector()

#Image Receiver
net_recvd_length = 0
recvd_image = b''

#Test Controller
direction = -1
cnt = 0

try:
    cap = cv2.VideoCapture(0)
except:
    print("webcam/video could not be loaded")
import time
while True:
    return_value, frame = cap.read()

    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #cv2.waitKey(1)
        net_recvd_length = 0
        recvd_image = b''

        #######################
        # Detect
        #######################
        bbox, bbox_label = detector.forward(frame)

        if bbox_label:
            print("BBOX: {}".format(bbox))
            print("BBOX_label: {}".format(bbox_label))
        else:
            print("False")

        # https://pymotw.com/3/socket/binary.html
        values = (bbox[0], bbox[1], 10, 10, float(bbox_label[0]))

        cv2.rectangle(frame, (int(bbox[0]), int(bbox[0])+10), (int(bbox[1]), int(bbox[1])+10), 1, 2)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 

    
    
        cv2.imshow('YoloV5 + Deep Sort', result)

        if cv2.waitKey(10) & 0xFF==ord('q'):
            break