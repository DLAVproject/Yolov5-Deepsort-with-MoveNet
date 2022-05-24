import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from PIL import Image

import torch
import torch.nn.functional as F

import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as util
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import math

from detector_helper import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Detector(object):
    def __init__(self):
        #load Yolo for object detection
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom
        # yolov5 model settings:
        self.model.classes = [0] # only detect humans -> class = 0
        self.model.max_det = 6
        self.model.conf = 0.5

        #Pose Classification Model
        self.interpreter = tf.lite.Interpreter(model_path='data/lite-model_movenet_singlepose_lightning_3.tflite')
        self.interpreter.allocate_tensors()

        # tracker
        #Definition of the parameters
        max_cosine_distance = 0.2 #The matching threshold. Samples with larger distance are considered an invalid match.
        nn_budget = 200 #[int] If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
        self.nms_max_overlap = 1.0 # ROIs that overlap more than this values are suppressed.(non_max_suppression)

        #Parameters for the Tracker():
        max_age = 500 #Maximum number of missed misses before a track is deleted.
        n_init = 15
        max_iou_distance = 0.7

        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        self.encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.tracker = Tracker(metric,max_iou_distance,max_age, n_init)
        

        # more parameters...
        self.initialized = False
        self.frame_num = 0
        self.in_frame = False

        self.first_pass = True
        self.pose_really_detected = False

        self.pose_number_of_frames = 20

        self.trigger_id = None
        self.pose_dict = {} # keys: track.track_id | values:np.array with poses saved --- idea: create dict with numpy arrays(key = track.track_id and add a new (key, value)-pair when a new person is detected 

    
    def forward(self, frame):
        
        self.frame_num += 1
        results = self.model(frame)
        #frame = results.render()[0]
    
        boxes, scores, names, imglist = get_bbox(results, frame)        
        bboxes = format_boxes(boxes.transpose()) # certain format of bboxes for the tracker
        
        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
        
        #initizialize color map
        cmap = plt.get_cmap('tab20b')    
        
        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]             
            
        #call the tracker
        self.tracker.predict()
        self.tracker.update(detections)        
        self.in_frame = False
        
        # update tracks
        
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
                
            bbox = track.to_tlbr()#sometimes the bounding boxes were negative
            _, __, bbox_width, bbox_height = track.to_tlwh() #used to get the width and the height of the bbox --> compared against threshold --> neglect small bounding boxes    
            class_name = track.get_class()   

            #img, label = get_pose_from_image_and_bounding_box(bbox, frame, bbox_width, bbox_height, self.interpreter)
            
            if self.initialized == False:
                #pose estimation:
                img, label = get_pose_from_image_and_bounding_box(bbox, frame, bbox_width, bbox_height, self.interpreter)

                #create 1 window for each tracked person
                window_name = 'Pose_Landmarks of person ' + str(track.track_id)
                cv2.imshow(window_name, img)

                if track.track_id not in self.pose_dict: #create a new empty (key,array) - pair in the dict
                    self.pose_dict[track.track_id] = np.zeros(self.pose_number_of_frames, dtype=object)
            
                self.pose_dict[track.track_id] = np.append(self.pose_dict[track.track_id],label)
                self.pose_dict[track.track_id] = np.delete(self.pose_dict[track.track_id],0) #delete the first item of the list --> keep the list short
            
                if(np.all(self.pose_dict[track.track_id] == "power to the people")): #if somebody holds the pose for certain frames set the boolean true
                    self.pose_really_detected = True
            
                #saving the POI ID
                if label == 'power to the people' and self.first_pass == True and self.pose_really_detected == True:
                    self.trigger_id = track.track_id            
                    self.tracker.tracks = [t for t in self.tracker.tracks if (t.track_id==self.trigger_id)]
                    self.first_pass = False 
                    self.initialized = True
                    cv2.destroyAllWindows()
                    
            if track.track_id == self.trigger_id:
                #color = colors[int(track.track_id) % len(colors)]
                #color = [i * 255 for i in color]
                color = (255,0,0) #blue
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 4)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2) 
                self.in_frame = True
                
        if self.in_frame:
            return [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2], [1]
        else:
            return [frame.shape[1]/2, frame.shape[0]/2], [0] #center of frame