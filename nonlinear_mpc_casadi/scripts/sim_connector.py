#!/usr/bin/env python

import rospy
from race.msg import drive_param
from ackermann_msgs.msg import AckermannDriveStamped

import math

pub = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=5)

def vel_and_angle(data):
	
	msg = AckermannDriveStamped();
	msg.header.stamp = rospy.Time.now();
	msg.header.frame_id = "base_link";

	msg.drive.speed = data.velocity
	msg.drive.acceleration = 1
	msg.drive.jerk = 1
	msg.drive.steering_angle = data.angle
	msg.drive.steering_angle_velocity = 1

	pub.publish(msg)


def listener():
	rospy.init_node('sim_connect', anonymous=True)
	rospy.Subscriber('drive_parameters', drive_param, vel_and_angle)
	rospy.spin()


if __name__=="__main__":
	listener()
