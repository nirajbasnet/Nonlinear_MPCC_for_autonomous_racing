#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path, Odometry

pub = rospy.Publisher("pf/viz/inferred_pose", PoseStamped, queue_size=1)

racecar_pose = Pose()

def timer_callback(event):
    global racecar_pose
    msg = PoseStamped()
    msg.pose = racecar_pose
    pub.publish(msg)

# Gets the racecar pose from racecar_simulator odom topic and redirect to appropriate msg time required by mpcc node.
# This helps to create same pose topics for both F1/10 car and the simulator.
def robot_pose_update(data):
    global racecar_pose
    racecar_pose = data.pose.pose

if __name__ == "__main__":
    rospy.init_node("remap_simulator_pose")
    # Set the update rate
    rospy.Timer(rospy.Duration(.02), timer_callback) # 50hz
    # Set subscribers
    rospy.Subscriber("/odom", Odometry, robot_pose_update)
    rospy.spin()
