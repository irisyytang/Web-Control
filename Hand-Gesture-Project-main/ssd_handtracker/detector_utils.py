# Original Source: Victor Dibia
# https://github.com/victordibia/handtracking

import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
import label_map_util
from collections import defaultdict
import coordinates
# #import mouse
from pynput.mouse import Button, Controller
import numpy as np
import wx

coordinates.init()
mouse = Controller()
(camx, camy) = (320, 240)
(sx, sy) = (1280, 800)

detection_graph = tf.Graph()
sys.path.append("..")

# Score threshold for showing bounding boxes - feel free to tweak
_score_thresh = 0.3

# Change these to your directories
PATH_TO_CKPT = "frozen_inference_graphs/frozen_inference_graph_30k.pb"
PATH_TO_LABELS = "hand_label_map.pbtxt"

NUM_CLASSES = 1  # We only want to detect hands - don't change this!

# Loads label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loads frozen inference graph
def load_inference_graph():

    print("Loading hand frozen graph into memory...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    print("Hand inference graph loaded.")
    return detection_graph, sess


# Draws bounding boxes
def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    
    ##oldPosition = (0,0)

    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            # print("top:: ", top)

            # Light green bounding box
            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)

            # Compute center of bounding box
            cX = int((left+right)/2)
            cY = int((top+bottom)/2)
            print("(cX, cY): ", cX, cY)
            # mouse.position = (678, 634)
            #mouse move
            mouseLoc=(cX*sx/camx, cY*sy/camy)
            print("3.mouse location:", mouseLoc)
            #if((np.subtract(oldPosition,mouseLoc))[0]>5 or (np.subtract(oldPosition,mouseLoc))[1]>5):
            
            mouse.position = mouseLoc
            #oldPosition = mouse.position
            
            print("1. Mouse Moved to:", mouse.position)
            
            # while mouse.position!=mouseLoc:
            #     pass
            
            # Label center of bounding box
            cv2.circle(image_np, (cX, cY), 10, (255, 255, 255), -1)
            cv2.putText(image_np, f"Center: ({cX}, {cY})", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            


# Displays FPS in light green
def draw_fps_on_image(fps, image_np):
    cv2.putText(image_np, fps, (750, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# Generates scores and bounding boxes based on webcam input
def detect_objects(image_np, detection_graph, sess):

    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected
    # Associated score are required for confidence levels
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores) = sess.run(
        [detection_boxes, detection_scores],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)


# Code to thread reading camera input.
# Source: Adrian Rosebrock
# https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def size(self):
        # return size of the capture device
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
