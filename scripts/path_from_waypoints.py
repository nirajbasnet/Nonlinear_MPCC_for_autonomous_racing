#!/usr/bin/env python

import rospy
import csv
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from tf.transformations import quaternion_from_euler


def create_header(frame_id):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    return header

def heading(yaw):
    q = quaternion_from_euler(0, 0, yaw)
    return Quaternion(*q)

def read_waypoints_from_csv(filename):
    # Import waypoints.csv into a list (path_points)
    poses_waypoints = []
    if filename == '':
        raise ValueError('No any file path for waypoints file')
    with open(filename) as f:
        path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
    path_points = [(float(point[0]), float(point[1]), float(point[2])) for point in path_points]
    for point in path_points:
        header = create_header('map')
        waypoint = Pose(Point(float(point[0]), float(point[1]), 0), heading(0.0))
        poses_waypoints.append(PoseStamped(header,waypoint))
    return poses_waypoints


if __name__=="__main__":
    # Initialize node
    rospy.init_node("path_from_waypoints_publisher")
    waypoints_pub = rospy.Publisher('/waypoints_path', Path, queue_size=1)
    filename = rospy.get_param('~waypoints_filepath', '')
    path = Path()
    path.header = create_header('map')
    path.poses = read_waypoints_from_csv(filename)
    rate = rospy.Rate(0.1)
    while not rospy.is_shutdown():
        waypoints_pub.publish(path)
        rate.sleep()
