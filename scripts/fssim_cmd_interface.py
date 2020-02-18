#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import  PoseStamped, Quaternion, PolygonStamped
from nav_msgs.msg import Path,Odometry
from std_msgs.msg import Header
from ackermann_msgs.msg import AckermannDriveStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from fssim_common.msg import State
from fsd_common_msgs.msg import ControlCommand


class FSSIM_CMD_INTERFACE:
    def __init__(self):
        rospy.init_node('fssim_cmd_interface')
        sub_fssim_pose_topic = rospy.get_param('fssim_pose_topic', '/fssim/base_pose_ground_truth')
        sub_f110_vel_topic = rospy.get_param('f110_cmd_vel_topic', '/vesc/high_level/ackermann_cmd_mux/input/nav_0')

        pub_fssim_cmd_vel_topic = rospy.get_param('fssim_cmd_vel_topic','/control/pure_pursuit/control_command')
        pub_odom_topic = rospy.get_param('f110_odom_topic', '/vesc/odom')
        pub_pose_topic = rospy.get_param('f110_pose_topic','/pf/viz/inferred_pose')

        self.PUBLISH_RATE = rospy.get_param('odom_publish_rate',40)
        self.car_state = None
        self.cmd_vel_pub = rospy.Publisher(pub_fssim_cmd_vel_topic,ControlCommand,queue_size=1)
        self.odom_pub = rospy.Publisher(pub_odom_topic,Odometry,queue_size=1)
        self.pose_pub = rospy.Publisher(pub_pose_topic,PoseStamped,queue_size=1)

        rospy.Subscriber(sub_fssim_pose_topic, State, self.pf_pose_callback, queue_size=1)
        rospy.Subscriber(sub_f110_vel_topic, AckermannDriveStamped, self.f110_cmd_callback, queue_size=1)

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def pf_pose_callback(self,msg):
        '''get state of the car'''
        self.car_state = msg

    def f110_cmd_callback(self,msg):
        '''get ackerman msg from f110 car'''
        fssim_cmd = ControlCommand()
        fssim_cmd.header = self.create_header('')
        fssim_cmd.throttle.data = msg.drive.acceleration
        fssim_cmd.steering_angle.data = msg.drive.steering_angle
        self.cmd_vel_pub.publish(fssim_cmd)

    def puhlish_odom_data_to_f110(self):
        '''publish odometry data '''
        loop_rate = rospy.Rate(self.PUBLISH_RATE)
        while not rospy.is_shutdown():
            if self.car_state is not None:
                current_state= self.car_state
                #localized state
                posedata= PoseStamped()
                posedata.header =self.create_header('map')
                posedata.pose.position.x= current_state.x
                posedata.pose.position.y = current_state.y
                quat_data = quaternion_from_euler(0,0,current_state.yaw)
                posedata.pose.orientation.z =quat_data[2]
                posedata.pose.orientation.w= quat_data[3]
                self.pose_pub.publish(posedata)

                #odometry message
                cmd = Odometry()
                cmd.header = self.create_header('map')
                cmd.child_frame_id = 'base_link'
                cmd.twist.twist.linear.x = np.linalg.norm(np.array([current_state.vx,current_state.vy]))
                self.odom_pub.publish(cmd)

                loop_rate.sleep()


if __name__ == '__main__':
    ffsim_cmd_interface = FSSIM_CMD_INTERFACE()
    ffsim_cmd_interface.puhlish_odom_data_to_f110()
    rospy.spin()
