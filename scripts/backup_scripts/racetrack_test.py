#!/usr/bin/env python
import numpy as np
import time
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import csv
from casadi import *

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Twist
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header
from tf.transformations import quaternion_from_euler, euler_from_quaternion

'''THINGS TO NOTE
1) Smoothness value(float greater than 0) can be specified to make the path smooth without any sharp transitions.
2) Notion of right or left wall depends upon the direction of movement in the track.For e.g, if the track is travelled in the anticlockwise direction, 
then the outer wall becomes right wall and inner wall becomes left wall.
'''

class RacetrackGen:
    def __init__(self):
        rospy.init_node('racetrack_generator')
        self.CENTER_TRACK_FILENAME= './kelley_third_floor-centerline.csv'
        self.RIGHT_TRACK_FILENAME = './kelley_third_floor-outerwall.csv'
        self.LEFT_TRACK_FILENAME = './kelley_third_floor-innerwall.csv'
        self.SMOOTH_CENTER_TRACK_FILENAME= './centerline_waypoints.csv'
        self.OUTPUT_FILE_PATH = rospy.get_param('output_path','./test/')
        self.center_path = Path()
        self.center_tangent_pub = rospy.Publisher('/center_tangent',PoseStamped,queue_size=1)
        self.right_tangent_pub = rospy.Publisher('/right_tangent', PoseStamped, queue_size=1)
        self.left_tangent_pub = rospy.Publisher('/left_tangent', PoseStamped, queue_size=1)
        self.center_path_pub = rospy.Publisher('/center_path', Path, queue_size=1)
        self.right_path_pub = rospy.Publisher('/right_path', Path, queue_size=1)
        self.left_path_pub = rospy.Publisher('/left_path', Path, queue_size=1)
        self.last_left_index = 0
        self.last_right_index=0


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

    def do_spline_interpolation(self,pts,smooth_value,derivative_order=0):
        # u_test = np.arange(len(pts))
        u_test= self.get_arc_lengths(pts)
        start= time.time()
        tck, u = splprep(pts.T, u=u_test, s=smooth_value, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=derivative_order)
        print("time=",time.time()-start)
        plt.plot(pts[:,0], pts[:,1], 'ro')
        plt.plot(x_new, y_new, 'b--')
        plt.show()
        interp_points = np.concatenate((x_new.reshape((-1,1)),y_new.reshape((-1,1))),axis=1)
        return interp_points

    def get_spline_interpolation(self, pts, smooth_value, derivative_order=0):
        u_test = np.arange(len(pts))
        tck, u = splprep(pts.T, u=u_test, s=smooth_value, per=1)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=derivative_order)
        return x_new,y_new



    def publish_path(self,waypoints,publisher):
        #Visualize path derived from the given waypoints in the path
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

    def test(self):
        center_lane = self.read_waypoints_from_csv(self.CENTER_TRACK_FILENAME)
        right_lane = self.read_waypoints_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_from_csv(self.LEFT_TRACK_FILENAME)
        center_interp_path = self.do_spline_interpolation(center_lane,0.5)
        right_interp_path=self.do_spline_interpolation(right_lane,0.1)

        print center_interp_path.shape

        dx_center, dy_center= self.get_spline_interpolation(center_lane,0.5,1)
        d_center = np.arctan2(dy_center,dx_center)
        # d_center= d_center / np.linalg.norm(d_center, ord=2, axis=1, keepdims=True)
        print(d_center.shape)
        print(d_center*180/3.14)
        print(d_center[0],d_center[1])
        idx = 0
        while not rospy.is_shutdown():
            tempPose = PoseStamped()
            tempPose.header = self.create_header('map')
            print center_interp_path[idx,0],center_interp_path[idx,0],self.heading((float(d_center[idx])))
            tempPose.pose.position.x = float(center_interp_path[idx,0])
            tempPose.pose.position.y = float(center_interp_path[idx,1])
            tempPose.pose.orientation = self.heading(float(d_center[idx]))
            print(float(d_center[idx]) * 180 / 3.14)
            self.center_tangent_pub.publish(tempPose)
            self.publish_path(center_lane,self.center_path_pub)
            # self.publish_path(right_lane,self.right_path_pub)

            idx = idx+5
            if idx>=d_center.shape[0]:
                idx=0
            rospy.sleep(0.25)

    def test2(self):
        car_pos=np.array([[20.33,20.1]])
        self.last_search_index = 5
        self.SEARCH_WINDOW=10
        self.SEARCH_TOL = 1.0
        self.path_points = self.read_waypoints_from_csv(self.CENTER_TRACK_FILENAME)
        win_upper_value = (self.last_search_index + self.SEARCH_WINDOW) % self.path_points.shape[0]
        if self.last_search_index >= self.SEARCH_WINDOW:
            win_lower_value = self.last_search_index - self.SEARCH_WINDOW
        else:
            win_lower_value = self.last_search_index + self.path_points.shape[0] - self.SEARCH_WINDOW
        if win_upper_value > win_lower_value:
            search_waypoints = self.path_points[win_lower_value:win_upper_value, :]
        else:
            search_waypoints = np.concatenate(
                (self.path_points[win_lower_value:, :], self.path_points[0:win_upper_value, :]), axis=0)

        start_time = time.time()
        distances_array = np.linalg.norm(search_waypoints - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        if distances_array[min_dist_idx] > self.SEARCH_TOL:
            win_lower_value, win_upper_value = 0,self.path_points.shape[0]-1
            distances_array = np.linalg.norm(self.path_points - car_pos, axis=1)
            min_dist_idx = np.argmin(distances_array)
        self.last_search_index = min_dist_idx

        print "time=",time.time()-start_time
        print self.path_points.shape
        print win_lower_value, win_upper_value
        print search_waypoints
        print min_dist_idx

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
        a1= np.where(a1<0,a1+2*np.pi,a1)
        if a2<0:
            a2 = a2 + 2*np.pi
        delta = abs(a1 - a2)
        alternate_delta = 2*np.pi - delta
        min_delta = np.minimum(delta,alternate_delta)
        return min_delta

    def get_wall_point(self,last_wall_idx,wall_points,center_point,center_points_angles,idx):
        SORT_WINDOW = 40

        SORT_DISTANCES_WINDOW =900
        SORT_ANGLES_WINDOW = 100
        distances_array = np.linalg.norm(wall_points - center_point, axis=1)
        sorted_distances_idx = np.argsort(distances_array)[0:SORT_DISTANCES_WINDOW]
        point_delta = wall_points[sorted_distances_idx] - center_point
        point_angle = np.arctan2(point_delta[:, 1], point_delta[:, 0])
        angle_delta = self.calculate_delta_angle(point_angle, center_points_angles[idx])
        sorted_delta_angle_idx = np.argsort(np.abs(angle_delta - np.pi / 2))[0:SORT_ANGLES_WINDOW]
        angle_idx=0
        desired_idx = sorted_distances_idx[sorted_delta_angle_idx[angle_idx]]
        if idx==0:
            last_wall_index=desired_idx
            print angle_idx,last_wall_index
        else:
            while 1:
                last_wall_idx_f=last_wall_idx
                if desired_idx<10 and last_wall_idx>wall_points.shape[0]-10:
                    desired_idx=desired_idx+wall_points.shape[0]
                elif desired_idx>wall_points.shape[0]-100 and last_wall_idx<100:
                    last_wall_idx_f = last_wall_idx+wall_points.shape[0]
                if abs(desired_idx-last_wall_idx_f)>200:
                    print angle_idx, sorted_delta_angle_idx[
                        angle_idx],desired_idx,last_wall_idx
                    # input('str1')
                    angle_idx +=1
                    desired_idx = sorted_distances_idx[sorted_delta_angle_idx[angle_idx]]
                else:
                    break
            last_wall_index = desired_idx % wall_points.shape[0]
        return wall_points[last_wall_index,:],point_angle[sorted_delta_angle_idx[angle_idx]],last_wall_index

        # point_delta = wall_points - center_point
        # point_angle = np.arctan2(point_delta[:, 1], point_delta[:, 0])
        # angle_delta = self.calculate_delta_angle(point_angle, center_points_angles[idx])
        # sorted_delta_angle_idx = np.argsort(np.abs(angle_delta - np.pi / 2))[0:SORT_WINDOW]
        # valid_points = wall_points[sorted_delta_angle_idx]
        # distances_array = np.linalg.norm(valid_points - center_point, axis=1)
        # min_dist_idx = np.argmin(distances_array)
        # desired_idx = sorted_delta_angle_idx[min_dist_idx]
        # return wall_points[desired_idx, :], point_angle[desired_idx],0

    # def find_nearest_index(self, car_pos):
    #     win_upper_value = (self.last_search_index + self.SEARCH_WINDOW) % self.center_lane.shape[0]
    #     print "start"
    #     print win_upper_value
    #     if self.last_search_index >= self.SEARCH_WINDOW:
    #         win_lower_value = self.last_search_index - self.SEARCH_WINDOW
    #     else:
    #         win_lower_value = self.last_search_index + self.center_lane.shape[0] - self.SEARCH_WINDOW
    #     print win_lower_value
    #     if win_upper_value > win_lower_value:
    #         search_waypoints = self.center_lane[win_lower_value:win_upper_value, :]
    #     else:
    #         search_waypoints = np.concatenate(
    #             (self.center_lane[win_lower_value:, :], self.center_lane[0:win_upper_value, :]), axis=0)
    #     distances_array = np.linalg.norm(search_waypoints - car_pos, axis=1)
    #     min_dist_idx = np.argmin(distances_array)
    #     if distances_array[min_dist_idx] > self.SEARCH_TOL:
    #         win_lower_value, win_upper_value = 0, self.center_lane.shape[0] - 1
    #         start=time.time()
    #         distances_array = np.linalg.norm(self.center_lane - car_pos, axis=1)
    #
    #         min_dist_idx = np.argmin(distances_array)
    #         print time.time()-start
    #     self.last_search_index = min_dist_idx
    #     print min_dist_idx, distances_array[min_dist_idx], win_lower_value, win_upper_value
    #     input('find_nearest_index')
    #     return min_dist_idx, distances_array[min_dist_idx], win_lower_value, win_upper_value

    def racetrack_generator(self):
        #Read waypoints from respective files
        center_lane = self.read_waypoints_from_csv(self.CENTER_TRACK_FILENAME)
        right_lane = self.read_waypoints_from_csv(self.RIGHT_TRACK_FILENAME)
        left_lane = self.read_waypoints_from_csv(self.LEFT_TRACK_FILENAME)

        # Interpolate center line upto desired resolution
        center_interp_path = self.get_interpolated_path(center_lane,smooth_value=0.75,scale=3,derivative_order=0)
        center_derivative = self.get_interpolated_path(center_lane,smooth_value=0.75,scale=3,derivative_order=1)
        center_points_angles= np.arctan2(center_derivative[:,1], center_derivative[:,0])

        #Interpolate outer and inner wall line upto 2X scale compared to center line
        right_interp_path = self.get_interpolated_path(right_lane,smooth_value=0.05,scale=30,derivative_order=0)
        left_interp_path = self.get_interpolated_path(left_lane, smooth_value=0.05, scale=30, derivative_order=0)

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
            centerPose = PoseStamped()
            centerPose.header = self.create_header('map')
            centerPose.pose.position.x = float(center_interp_path[idx, 0])
            centerPose.pose.position.y = float(center_interp_path[idx, 1])
            centerPose.pose.orientation = self.heading(float(center_points_angles[idx]))
            self.center_tangent_pub.publish(centerPose)

            rightPose = PoseStamped()
            rightPose.header = self.create_header('map')
            rightPose.pose.position.x = float(final_right_path[idx, 0])
            rightPose.pose.position.y = float(final_right_path[idx, 1])
            rightPose.pose.orientation = self.heading(float(final_right_point_angle[idx]))
            self.right_tangent_pub.publish(rightPose)

            leftPose = PoseStamped()
            leftPose.header = self.create_header('map')
            leftPose.pose.position.x = float(final_left_path[idx, 0])
            leftPose.pose.position.y = float(final_left_path[idx, 1])
            leftPose.pose.orientation = self.heading(float(final_left_point_angle[idx]))
            self.left_tangent_pub.publish(leftPose)


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

    def test_casadi_interpolation(self):
        center_lane = self.read_waypoints_from_csv(self.SMOOTH_CENTER_TRACK_FILENAME)
        angle_data = self.read_waypoints_from_csv('./center_spline_derivatives.csv')
        # u = np.arange(center_lane.shape[0])
        # V_X = center_lane[:,0]
        # V_Y = center_lane[:,1]
        # lut_center_x = interpolant('LUT_center_x', 'bspline', [u], V_X)
        # lut_center_y = interpolant('LUT_center_y', 'bspline', [u], V_Y)
        # u_new = np.linspace(u.min(), u.max(), 1000)
        # x_new= lut_center_x(u_new)
        # y_new= lut_center_y(u_new)
        # center_interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
        # plt.plot(center_lane[:, 0], center_lane[:, 1], 'ro')
        # plt.plot(x_new, y_new, 'b--')
        # # plt.show()
        # u2 = np.arange(angle_data.shape[0])
        # ax = angle_data[:, 0]
        # ay=  angle_data[:, 1]
        # lut_center_dx = interpolant('LUT_center_dx', 'bspline', [u2], ax)
        # lut_center_dy = interpolant('LUT_center_dy', 'bspline', [u2], ay)
        # u2_new = np.linspace(u2.min(), u2.max(), 1000)
        # d_center_X = lut_center_dx(u2_new)
        # d_center_Y= lut_center_dy(u2_new)

        u= self.get_arc_lengths(center_lane)
        lut_center_x,lut_center_y = self.get_interpolated_path_casadi('lut_center_x','lut_center_y',center_lane,u)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new= lut_center_x(u_new)
        y_new= lut_center_y(u_new)
        center_interp_points = np.concatenate((x_new.reshape((-1, 1)), y_new.reshape((-1, 1))), axis=1)
        lut_center_dx, lut_center_dy = self.get_interpolated_path_casadi('lut_center_dx', 'lut_center_dy', angle_data, u)
        d_center_X = lut_center_dx(u_new)
        d_center_Y= lut_center_dy(u_new)
        center_points_angles = np.arctan2(d_center_Y,d_center_X)
        idx=0
        x = MX.sym('x')
        f = Function('f', [x], [lut_center_x(x),lut_center_y(x)] )
        s = MX.sym('s')
        out= Function('out',[x],[lut_center_x(x)])
        test = external('test','./out.so')
        a=time.time()
        for i in range(100):
            b=out(i/10.0)
        print time.time()-a, "in"
        cc = time.time()
        # for i in range(100):
        #     b = test(i / 10.0)
        # print time.time() - cc, "in"

        print out(1.5),test(1.5)
        print lut_center_x(1.5),lut_center_y(1.5)

        # while not rospy.is_shutdown():
        #     centerPose = PoseStamped()
        #     centerPose.header = self.create_header('map')
        #     centerPose.pose.position.x = float(center_interp_points[idx, 0])
        #     centerPose.pose.position.y = float(center_interp_points[idx, 1])
        #     centerPose.pose.orientation = self.heading(float(center_points_angles[idx]))
        #     self.center_tangent_pub.publish(centerPose)
        #
        #     print idx,float(center_points_angles[idx])*180/3.14
        #     idx = idx + 1
        #     if idx >= center_interp_points.shape[0]:
        #         idx = 0
        #     rospy.sleep(0.02)

    def get_interpolated_path_casadi(self,label_x,label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = interpolant(label_x, 'bspline', [u], V_X)
        lut_y = interpolant(label_y, 'bspline', [u], V_Y)
        return lut_x,lut_y



if __name__=="__main__":
    racetrack_gen = RacetrackGen()
    # racetrack_gen.test()
    racetrack_gen.racetrack_generator()

    # racetrack_gen.test_casadi_interpolation()
    rospy.spin()
