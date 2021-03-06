# Original Source: Victor Dibia
# https://github.com/victordibia/handtracking

import argparse
import cv2
import datetime
import detector_utils
from detector_utils import WebcamVideoStream
import multiprocessing
from multiprocessing import Queue, Pool
import tensorflow as tf
import time
import coordinates
from pynput.mouse import Button, Controller
import numpy as np
import wx



# app=wx.App(False)
# (sx,sy)=wx.GetDisplaySize()
# print('sx, sy :: ', sx, sy)


frame_processed = 0
score_thresh = 0.3
fps_values = []

# try:
#     del app
# except:
#     pass


# Creates a worker thread that loads graph and does detection on images in an input queue, then puts it on an output queue
# This is the backbone for the multi-threading approach
def worker(input_q, output_q, cap_params, frame_processed):
    print(">> Loading frozen model for worker.")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.compat.v1.Session(graph=detection_graph)
    while True:
        frame = input_q.get()
        if (frame is not None):
            '''Boxes contain coordinates for detected hands
            Scores contains confidence levels
            If len(boxes) > 1, at least one hand is detected
            You can change the score_thresh value as desired'''
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draws bounding boxes
            detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            

            # Adds frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            output_q.put(frame)
    sess.close()


# Prints average fps
def average_fps():
    total_fps = 0
    for fps in fps_values:
        total_fps += fps
    print(f"Average FPS: {total_fps/len(fps_values)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=1,  # Change this for max number of hands to detect
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=800,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=450,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # Max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # Spins up workers for parallel detection
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    #start_time = datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow("Hand Tracker, SSD + Multi-Threaded Detection",
                    cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        ## Iris added lines
        w=np.shape(frame)[1]
        h=np.shape(frame)[0]
        frame=frame[1:h-199,250:w].copy()

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        fps_values.append(fps)

        # average_fps() # Feel free to run this if you wanna check the fps as the program runs

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS: " + str(int(fps)),
                                                     output_frame)

                cv2.imshow(
                    "Hand Tracker, SSD + Multi-Threaded Detection", output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("Frames Processed: ", index, "Elapsed Time: ",
                          elapsed_time, "FPS: ", str(int(fps)))
        else:
            break

        ######## build up mouse ######
        # mouseLoc=(sx-(detector_utils.cX*sx/camx), detector_utils.cY*sy/camy)
        
        # mouse.position=mouseLoc 
        # while mouse.position!=mouseLoc:
        #     pass


        ######## click action ######

        # if angle<15:
            
        #     mouse.click(button='left');    # uncomment to activate the left click
        #         print('left clicked');
        #         pass
        #     else:
        #         pass




        # Esc to quit program
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break


    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("FPS", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
