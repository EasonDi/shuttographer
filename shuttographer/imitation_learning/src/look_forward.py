#!/usr/bin/env python3
import sys
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState

class LookForwardNode():
    """Node that makes Shutter's wrist_1_link rotate to make the Zed camera look forward."""
    def __init__(self):
        # Init the node
        rospy.init_node('look_forward', anonymous=True)

        # publish rate
        publish_rate = 1 # Hz

        # parameters for the joint of interest
        self.joint_name = "joint_4"
        self.desired_joint_position = 0.0
        self.joint_reached_desired_position = False
        self.joint_command = [0.0, 0.0, 0.0, 0.0]

        # Publishers
        self.joint_pub = rospy.Publisher("/unity_joint_group_controller/command", Float64MultiArray, queue_size=5)

        # Subscribers
        self.joints_sub = rospy.Subscriber("/joint_states", JointState, self.joints_callback, queue_size=5)

        rate = rospy.Rate(publish_rate)
        while not rospy.is_shutdown():
            if not self.joint_reached_desired_position:
                msg = Float64MultiArray()
                self.joint_command[3] = self.desired_joint_position
                msg.data = self.joint_command
                self.joint_pub.publish(msg)
            else:
                break # end the execution of the node
            rate.sleep()


    def joints_callback(self, msg):
        # current joint position
        self.joint_command = msg.position
        joint_position = msg.position[3]
        rospy.logdebug(f'joint position: {joint_position}')

        if np.fabs(joint_position - self.desired_joint_position) < 1e-2:
            self.joint_reached_desired_position = True
        else:
            self.joint_reached_desired_position = False
        rospy.logdebug(f'reached? {self.joint_reached_desired_position}')


if __name__ == '__main__':
    try:
        node = LookForwardNode()
    except rospy.ROSInterruptException:
        pass

    sys.exit(0)