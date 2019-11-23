#!/usr/bin/env python
import numpy as np
import time
import os
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
from casadi import *

import rospy
from geometry_msgs.msg import  PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion

'''THINGS TO NOTE
1) Smoothness value(float greater than 0) can be specified to make the path smooth without any sharp transitions.
2) Notion of right or left wall depends upon the direction of movement in the track.For e.g, if the track is travelled in the anticlockwise direction, 
then the outer wall becomes right wall and inner wall becomes left wall.
'''


class RacetrackGen:
    def __init__(self):
        rospy.init_node('racetrack_generator')
        dirname = os.path.dirname(__file__)
        folder_name = rospy.get_param('path_folder_name', 'kelley')
        folder_path = 'raw_track_data/'+ folder_name+'/'
        self.CENTER_TRACK_FILENAME = os.path.join(dirname,folder_path + folder_name+'-centerline.csv')
        self.RIGHT_TRACK_FILENAME = os.path.join(dirname, folder_path + folder_name + '-outerwall.csv')
        self.LEFT_TRACK_FILENAME = os.path.join(dirname, folder_path+ folder_name + '-innerwall.csv')
        self.OUTPUT_FILE_PATH = rospy.get_param('output_path','./generated_tracks/')
        self.center_path = Path()
        self.center_tangent_pub = rospy.Publisher('/center_tangent',PoseStamped,queue_size=1)
        self.right_tangent_pub = rospy.Publisher('/right_tangent', PoseStamped, queue_size=1)
        self.left_tangent_pub = rospy.Publisher('/left_tangent', PoseStamped, queue_size=1)
        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=1)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=1)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=1)
        self.last_left_index = 0
        self.last_right_index=0
        self.CENTER_SMOOTH_VALUE =rospy.get_param('center_smooth_value', 0.75)
        self.CENTER_SCALING = rospy.get_param('center_scaling', 3)
        self.LEFT_SMOOTH_VALUE = rospy.get_param('left_smooth_value', 0.05)
        self.LEFT_SCALING = rospy.get_param('left_scaling', 30)
        self.RIGHT_SMOOTH_VALUE = rospy.get_param('right_smooth_value', 0.05)
        self.RIGHT_SCALING = rospy.get_param('right_scaling', 30)
        self.sort_dist_window_percent=12.5
        self.sort_angle_window_percent=1.35
        self.DEBUG_MODE = False
        self.REVERSE_MOVING_DIRECTION = False  #can select clockwise or anticlockwise direction

    def read_waypoints_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of numpy array'''
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = np.array([[float(point[0]), float(point[1])] for point in path_points])
        return path_points

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return yaw

    def get_interpolated_path(self,pts,smooth_value,scale=2,derivative_order=0):
        u_test = np.arange(len(pts))
        tck, u = splprep(pts.T, u=u_test, s=smooth_value, per=1)
        u_new = np.linspace(u.min(), u.max(), len(pts)*scale)
        x_new, y_new = splev(u_new, tck, der=derivative_order)
        interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
        return interp_points

    def vec_angle(self,v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

    def calculate_delta_angle(self,a1,a2):
        """ Calculate"""
        a1= np.where(a1<0,a1+2*np.pi,a1)
        if a2<0:
            a2 = a2 + 2*np.pi
        delta = abs(a1 - a2)
        alternate_delta = 2*np.pi - delta
        min_delta = np.minimum(delta,alternate_delta)
        return min_delta

    def get_wall_point(self,last_wall_idx,wall_points,center_point,center_points_angles,idx):
        SORT_DISTANCES_WINDOW =int(self.sort_dist_window_percent*len(wall_points)/100.0)
        SORT_ANGLES_WINDOW = int(self.sort_angle_window_percent*len(wall_points)/100.0)
        distances_array = np.linalg.norm(wall_points - center_point, axis=1)
        #sort points based on distance to the center line point and pick points equal to SORT_DISTANCES_WINDOW
        sorted_distances_idx = np.argsort(distances_array)[0:SORT_DISTANCES_WINDOW]
        point_delta = wall_points[sorted_distances_idx] - center_point
        point_angle = np.arctan2(point_delta[:, 1], point_delta[:, 0])
        #finding difference in angle between the tangent at centerline and line joining wall point and centerline point
        angle_delta = self.calculate_delta_angle(point_angle, center_points_angles[idx])
        # sorting based on proximity of delta angle to 90 degrees which is perpendicular condition
        sorted_delta_angle_idx = np.argsort(np.abs(angle_delta - np.pi / 2))[0:SORT_ANGLES_WINDOW]
        angle_idx=0
        desired_idx = sorted_distances_idx[sorted_delta_angle_idx[angle_idx]]
        if idx==0:
            last_wall_index=desired_idx
            if self.DEBUG_MODE:
                print angle_idx,last_wall_index
        else:
            while 1:
                last_wall_idx_f=last_wall_idx
                #checking for end of the loop case and adjusting to find exact delta between consecutive final wall points
                if desired_idx<10 and last_wall_idx>wall_points.shape[0]-10:
                    desired_idx=desired_idx+wall_points.shape[0]
                elif desired_idx>wall_points.shape[0]-SORT_ANGLES_WINDOW and last_wall_idx<SORT_ANGLES_WINDOW:
                    last_wall_idx_f = last_wall_idx+wall_points.shape[0]
                #making sure consecutive points are close to each other
                #if points found are not close to each other then next nearest delta angle point is selected and checked again
                if abs(desired_idx-last_wall_idx_f)>(2*SORT_ANGLES_WINDOW):
                    if self.DEBUG_MODE:
                        print angle_idx, sorted_delta_angle_idx[
                            angle_idx],desired_idx,last_wall_idx
                    angle_idx +=1
                    desired_idx = sorted_distances_idx[sorted_delta_angle_idx[angle_idx]]
                else:
                    break
            last_wall_index = desired_idx % wall_points.shape[0]
        return wall_points[last_wall_index,:],point_angle[sorted_delta_angle_idx[angle_idx]],last_wall_index

    def publish_path(self, waypoints, publisher):
        # Visualize path derived from the given waypoints in the path
        path = Path()
        path.header = self.create_header('map')
        path.poses = []
        for point in waypoints:
            tempPose = PoseStamped()
            tempPose.header = self.center_path.header
            tempPose.pose.position.x = point[0]
            tempPose.pose.position.y = point[1]
            tempPose.pose.orientation.w = 1.0
            path.poses.append(tempPose)
        publisher.publish(path)

    def publish_tangent(self,publisher,x,y,yaw):
        #Visualize tangent at a point in the path
        t_pose = PoseStamped()
        t_pose.header = self.create_header('map')
        t_pose.pose.position.x = x
        t_pose.pose.position.y = y
        t_pose.pose.orientation = self.heading(yaw)
        publisher.publish(t_pose)

    def racetrack_generator(self):
        #Read waypoints from respective files
        center_lane = self.read_waypoints_from_csv(self.CENTER_TRACK_FILENAME)
        right_lane = self.read_waypoints_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_from_csv(self.LEFT_TRACK_FILENAME)
        if self.REVERSE_MOVING_DIRECTION:
            center_lane = center_lane[::-1]

        # Interpolate center line upto desired resolution
        center_interp_path = self.get_interpolated_path(center_lane,smooth_value=self.CENTER_SMOOTH_VALUE,scale=self.CENTER_SCALING,derivative_order=0)
        center_derivative = self.get_interpolated_path(center_lane,smooth_value=self.CENTER_SMOOTH_VALUE,scale=self.CENTER_SCALING,derivative_order=1)
        center_points_angles= np.arctan2(center_derivative[:,1], center_derivative[:,0])

        #Interpolate outer and inner wall line upto 2X scale compared to center line
        right_interp_path = self.get_interpolated_path(right_lane,smooth_value=self.RIGHT_SMOOTH_VALUE,scale=self.RIGHT_SCALING,derivative_order=0)
        left_interp_path = self.get_interpolated_path(left_lane, smooth_value=self.LEFT_SMOOTH_VALUE, scale=self.LEFT_SCALING, derivative_order=0)

        final_right_path = np.zeros_like(center_interp_path)
        final_left_path = np.zeros_like(center_interp_path)
        final_right_path_width = np.zeros_like(center_points_angles)
        final_left_path_width = np.zeros_like(center_points_angles)

        #Perpendicular angle satisfying criteria for wall points
        final_right_point_angle=np.zeros_like(center_points_angles)
        final_left_point_angle = np.zeros_like(center_points_angles)

        for idx,center_point in enumerate(center_interp_path):

            final_right_path[idx,:],final_right_point_angle[idx],self.last_right_index = self.get_wall_point(self.last_right_index,right_interp_path,center_point,center_points_angles,idx)
            final_left_path[idx, :], final_left_point_angle[idx],self.last_left_index = self.get_wall_point(self.last_left_index,left_interp_path,
                                                                                         center_point,
                                                                                         center_points_angles, idx)
            final_right_path_width[idx]=np.linalg.norm(final_right_path[idx,:]-center_interp_path[idx,:])
            final_left_path_width[idx] = np.linalg.norm(final_left_path[idx, :] - center_interp_path[idx, :])
        trackwidths = np.column_stack((final_right_path_width,final_left_path_width))

        #Visualization code
        self.publish_path(center_interp_path, self.center_path_pub)
        self.publish_path(right_interp_path, self.right_path_pub)
        self.publish_path(left_interp_path, self.left_path_pub)

        idx = 0
        element_arc_lengths = self.get_arc_lengths(center_interp_path)
        element_arc_lengths = np.column_stack((element_arc_lengths,center_points_angles))
        np.savetxt(self.OUTPUT_FILE_PATH+'centerline_waypoints.csv', center_interp_path, delimiter=",")
        np.savetxt(self.OUTPUT_FILE_PATH + 'right_waypoints.csv', final_right_path, delimiter=",")
        np.savetxt(self.OUTPUT_FILE_PATH + 'left_waypoints.csv', final_left_path, delimiter=",")
        np.savetxt(self.OUTPUT_FILE_PATH + 'track_widths.csv', trackwidths, delimiter=",")
        np.savetxt(self.OUTPUT_FILE_PATH + 'center_spline_derivatives.csv', center_derivative, delimiter=",")

        while not rospy.is_shutdown():
            self.publish_tangent(self.center_tangent_pub,center_interp_path[idx, 0],center_interp_path[idx, 1],center_points_angles[idx])
            self.publish_tangent(self.right_tangent_pub, final_right_path[idx, 0], final_right_path[idx, 1],
                                 final_right_point_angle[idx])
            self.publish_tangent(self.left_tangent_pub, final_left_path[idx, 0], final_left_path[idx, 1],
                                 final_left_point_angle[idx])
            if self.DEBUG_MODE:
                print "right",center_points_angles[idx],final_right_point_angle[idx],(abs(center_points_angles[idx]-final_right_point_angle[idx]))%(np.pi/2),final_right_path_width[idx]
                print "left",center_points_angles[idx], final_right_point_angle[idx], (abs(
                    center_points_angles[idx] - final_right_point_angle[idx])) % (np.pi / 2) , final_left_path_width[idx]
            idx = idx + 1
            if idx >= center_interp_path.shape[0]:
                idx = 0
            rospy.sleep(0.25)

    def get_arc_lengths(self,waypoints):
        d = np.diff(waypoints,axis=0)
        consecutive_diff= np.sqrt(np.sum(np.power(d, 2), axis=1))
        dists_cum = np.cumsum(consecutive_diff)
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum

if __name__=="__main__":
    racetrack_gen = RacetrackGen()
    racetrack_gen.racetrack_generator()
    rospy.spin()
