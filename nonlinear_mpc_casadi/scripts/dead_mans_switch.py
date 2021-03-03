#!/usr/bin/env python

import rospy
import pdb
import numpy as np
import tf
import os
import time
from race.msg import drive_param
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive
from std_msgs.msg import Header,Bool


pub = rospy.Publisher("vesc/high_level/ackermann_cmd_mux/input/nav_0",
                              AckermannDriveStamped,
                              queue_size = 10)
reset_pub= rospy.Publisher("pure_pursuit_reset",Bool,queue_size=1)


JOY_TIMER_PERIOD = 100000000   # 100 million nanoseconds -> 0.1 seconds
JOY_INIT_DELAY = 2             # 5 seconds

joystick_present = False
timer_callback_running = True

# Callback for monitoring whether the joystick is present or not.
# Sets joystick_present to prevent the car from going crazy when the joystick
# gets disconnected or is not present.
# Runs every 1/10 second.
def timer_callback(event):
    global joystick_present
    global joy_timer

    # Need a local (new_joystick_present) since without it, we would keep delaying as long
    # as the joystick is plugged in.
    # This could be bad, for example, if we unplugged the joystick while the delay is being
    # executed, as the car would go berserk.
    new_joystick_present = False
    for root, dirs, files in os.walk("/dev/input"):
        if "js0" in files:
            new_joystick_present = True
            break

    # The joystick was reconnected: delay for JOY_INIT_DELAY seconds to give ROS time
    # to re-initialize it. Otherwise, the car might go on its own without us holding the right bumper.
    if new_joystick_present and not joystick_present:
        reset_pub.publish(Bool(True))
        time.sleep(JOY_INIT_DELAY)

    joystick_present = new_joystick_present
    
    
    joy_timer = rospy.timer.Timer(rospy.Duration(0, JOY_TIMER_PERIOD), timer_callback, oneshot=True)

# Need to initialize joy_timer twice: once at startup (here), and another time in the callback.
# Reason: the time.sleep() call introduces delay and might lead to the callback being called again when
# it should be sleeping. This is also why we must use a one-shot timer instead of a repeating one.
joy_timer = rospy.timer.Timer(rospy.Duration(0, JOY_TIMER_PERIOD), timer_callback, oneshot=True)

def callback(data):
    global joystick_present
    global joy_timer

    # Set velocity and angle to 0.0 unless the joystick is present.
    velocity = 0.0
    angle = 0.0
    
    if joystick_present:
        velocity = data.velocity
        angle = data.angle
    
    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = rospy.Time.now()
    drive_msg.header.frame_id = "base_link"
    drive_msg.drive.steering_angle = angle
    drive_msg.drive.speed = velocity
    pub.publish(drive_msg)

if __name__ == "__main__":
    global joy_timer
    rospy.init_node("dead_mans_switch")
    rospy.Subscriber("drive_parameters", drive_param, callback, queue_size=1)
    joy_timer.run()
    rospy.spin()
