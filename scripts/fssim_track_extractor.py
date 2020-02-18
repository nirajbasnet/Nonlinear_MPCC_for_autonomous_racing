#!/usr/bin/env python
import numpy as np
import time
import os
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv

import rospy
from geometry_msgs.msg import  PoseStamped, Quaternion, PolygonStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from fssim_common.msg import Track

'''
center_line_topic_name: /control/pure_pursuit/center_line    message type:geometry_msgs/PolygonStamped
track_cones_topic_name: /fssim/track                         message type:fssim_common/Track
'''
class FSSIM_Track:
    def __init__(self):
        rospy.init_node('fssim_track_extractor')
        dirname = os.path.dirname(__file__)
        folder_name = rospy.get_param('path_folder_name', 'fssim')
        folder_path = 'raw_track_data/' + folder_name + '/'
        self.CENTER_TRACK_FILENAME = os.path.join(dirname, folder_path + folder_name + '-centerline.csv')
        self.LEFT_TRACK_FILENAME = os.path.join(dirname, folder_path + folder_name + '-outerwall.csv')
        self.RIGHT_TRACK_FILENAME = os.path.join(dirname, folder_path + folder_name + '-innerwall.csv')


        center_line_topic_name = rospy.get_param('center_line_topic','/control/pure_pursuit/center_line')
        track_cones_topic_name = rospy.get_param('track_cones_topic','/fssim/track')
        rospy.Subscriber(center_line_topic_name,PolygonStamped, self.center_line_callback)
        rospy.Subscriber(track_cones_topic_name,Track,self.track_callback)
        self.center_line_data = None
        self.left_cones_data =None
        self.right_cones_data =None



    def center_line_callback(self,msg):
        '''centerline message in polygon format'''

        # print msg.polygon.points
        if self.center_line_data is None:
            self.center_line_data = np.array([[point.x, point.y] for point in msg.polygon.points])


    def track_callback(self,msg):
        '''left cones and right cones position in the track'''
        # print msg.cones_left
        if self.left_cones_data is None:
            self.left_cones_data = np.array([[point.x, point.y] for point in msg.cones_left])
            self.right_cones_data = np.array([[point.x, point.y] for point in msg.cones_right])


    def save_track_data_in_file(self):
        while not rospy.is_shutdown():
            if (self.center_line_data is not None) and (self.left_cones_data is not None):
                np.savetxt(self.CENTER_TRACK_FILENAME , self.center_line_data[::4], delimiter=",")
                np.savetxt(self.RIGHT_TRACK_FILENAME , self.right_cones_data, delimiter=",")
                np.savetxt(self.LEFT_TRACK_FILENAME , self.left_cones_data, delimiter=",")
                break

if __name__=="__main__":
    racetrack_gen = FSSIM_Track()
    racetrack_gen.save_track_data_in_file()








