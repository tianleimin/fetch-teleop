#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""
    Extract images from a rosbag (Python 2)
    Adapted from: https://gist.github.com/priteshgohil/c3cf492b5705cd5536a68b60a0e89c52
"""

from __future__ import print_function
import sys
import os
import argparse
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

# Print iterations progress
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, bar_length=50):
    """
    Call in a loop to create terminal progress bar
    Adapted from: https://stackoverflow.com/q/3173320
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = '{0:.' + str(decimals) + 'f}'
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\033[F')  # Move cursor up one line
    sys.stdout.write('\033[K')  # Clear the line after the cursor
    sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix))
    sys.stdout.flush()

# Print ROS bag info and save frames as images
def saveFrame(pid, camera):

    # choose image topic for the given camera
    if camera == 'oakd':
        image_topic = '/global_camera/compressed'
        fps = 12
    elif camera == 'fetch':
        image_topic = '/head_camera/rgb/image_rect_color/compressed'
        fps = 30
    else:
        print("Invalid value! default camera choice to oakd")
        image_topic = '/global_camera/compressed'
        fps = 12

    bag_file = str('bags/'+pid+'.bag') # input bag file
    output_dir = str('images_'+camera+'/'+pid+'/') # output directory
    print ("\nExtracting images from %s on topic %s into %s" % (bag_file, image_topic, output_dir))

    # Create output directory if does not exist
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Print ROS bag info
    bag = rosbag.Bag(bag_file, "r")
    bridge = CvBridge()
    print("\nDuration:", bag.get_end_time()-bag.get_start_time(),"sec")
    topics= bag.get_type_and_topic_info()[1].keys() # all the topics info
    print("\nAvailable topics in bag file are:{}".format(topics))
    total_frames = bag.get_message_count(image_topic) # Total frames in topic
    print("\nTotal frames in bag file are:{}\n".format(total_frames))

    """
    types=[] # info about each topic types
    for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
        types.append(bag.get_type_and_topic_info()[1].values())
    print("\nTypes are: {}".format(types))

    # for topic, msg, t in bag.read_messages(topics=[image_topic]):
        # print("Size of the image: W {} x H {}".format(msg.width, msg.height))
        # print("Encoding of the frames: {}".format(msg.encoding))
        # break
    """

    basename = os.path.splitext(os.path.basename(bag_file))[0]
    count = 0
    printProgressBar(0, total_frames, prefix='writing frames:'.ljust(15), suffix='Complete')

    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        # save 2 frames per second, otherwise too many images
        if count % (fps/2) == 0:
            try:
            	# Convert compressed image to RAW
            	cv_img = bridge.compressed_imgmsg_to_cv2(msg)
            	# save image file
            	p = os.path.join(output_dir, basename)
            	p = p + "_frame_{:06}".format(count)+".png"
            	cv2.imwrite(p, cv_img)
            except:
            	print ("issue saving frame %i, skipped" % count)
            	pass
        count += 1
        printProgressBar(count, total_frames, prefix='writing frames:'.ljust(15), suffix='Complete')

    bag.close()
    print("\nFinished processing %s on topic %s" % (bag_file, image_topic))

    return


# main loop to process all the bags
# list of participants and cameras
participants = ['p1_2022-07-25', 'p2_2022-08-03', 'p3_2022-08-04', 'p4_2022-08-10', 'p5_2022-08-12', 
                'p6_2022-08-15', 'p7_2022-08-15', 'p8_2022-08-19', 'p9_2022-08-22', 'p10_2022-08-26', 
                'p11_2022-08-29', 'p12_2022-08-29', 'p13_2022-08-29', 'p14_2022-08-31', 'p15_2022-09-02', 
                'p16_2022-09-05', 'p17_2022-09-06', 'p18_2022-09-07', 'p19_2022-09-08', 'p20_2022-09-09']
cameras = ['oakd', 'fetch']

# loop over all participants
for pid in participants:
    # extract frames for both camera feeds
    for camera in cameras:
        saveFrame(pid, camera)

