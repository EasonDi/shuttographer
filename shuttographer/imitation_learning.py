#!/usr/bin/env python3

import os
from datetime import datetime
import rospkg
import rospy
from sensor_msgs.msg import JointState
import tf2_ros
import tf2_geometry_msgs
import numpy as np
from urdf_parser_py.urdf import URDF
from visualization_msgs.msg import MarkerArray


class ExpertNode(object):
    """
    Node that simulates an expert controller using optimization. It controls two joints of the robot to make it
    point towards a target.
    """

    def __init__(self):
        rospy.init_node('expert_opt')
        rospy.on_shutdown(self.cleanup)

        # params
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.biceps_link = rospy.get_param("~biceps_link", "biceps_link")
        self.camera_link = rospy.get_param("~camera_link", "camera_color_optical_frame")
        self.save_state_actions = rospy.get_param("~save_state_actions", True)
        self.human_pose_topic = rospy.get_param('~human_pose_topic', '/body_tracking_data')

        if self.save_state_actions:
            rospack = rospkg.RosPack()
            bc_dir = rospack.get_path("shuttographer")
            default_output_file = os.path.join(bc_dir, "data", "state_actions.txt")
            self.output_file = rospy.get_param("~output_file", default_output_file)

            base_dir = os.path.dirname(self.output_file)
            if not os.path.isdir(base_dir):
                os.makedirs(base_dir)

            self.fid = open(self.output_file, 'a')  # open output buffer to record state-action pairs...
            date = datetime.now()
            self.fid.write("# data from {}\n".format(date.strftime("%d/%m/%Y %H:%M:%S")))

        else:
            self.fid = None

        # joint values
        self.current_pose = None #[0.0, 0.0, 0.0, 0.0]
        self.past_pose = None

        # get robot model
        self.robot = URDF.from_parameter_server()

        # tf subscriber
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # joint subscriber
        rospy.Subscriber('/joint_states', JointState, self.joints_callback, queue_size=5)
        #subscribe to human body joints
        rospy.Subscriber(self.human_pose_topic, MarkerArray, self.body_callback, queue_size=5)
        rospy.spin()

    def cleanup(self):
        """
        Be good with the environment.
        """
        if self.fid is not None:
            self.fid.close()

    def joints_callback(self, msg):
        """
        Joints callback
        :param msg: joint state
        """
        joint1_idx = -1
        joint2_idx = -1
        joint3_idx = -1
        joint4_idx = -1
        for i in range(len(msg.name)):
            if msg.name[i] == 'joint_1':
                joint1_idx = i
            elif msg.name[i] == 'joint_2':
                joint2_idx = i
            elif msg.name[i] == 'joint_3':
                joint3_idx = i
            elif msg.name[i] == 'joint_4':
                joint4_idx = i
        assert joint1_idx >= 0 and joint2_idx >= 0 and joint3_idx >= 0 and joint4_idx >= 0, \
            "Missing joints from joint state! joint1 = {}, joint2 = {}, joint3 = {}, joint4 = {}".\
                format(joint1_idx, joint2_idx, joint3_idx, joint4_idx)
        self.past_pose = [self.current_pose[0],
                          self.current_pose[1],
                          self.current_pose[2],
                          self.current_pose[3]]

        self.current_pose = [msg.position[joint1_idx],
                             msg.position[joint2_idx],
                             msg.position[joint3_idx],
                             msg.position[joint4_idx]]

    def body_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            rospy.logwarn("Joint positions are unknown. Waiting to receive joint states.")
            return

        trans = self.tf_buffer.lookup_transform('/rgb_camera_link', 'base_footprint', rospy.Time(0), rospy.Duration(1))
        closest_joint = []
        #track the closest body joint
        for marker in msg.markers:
            #track the nose position
            if marker.id % 100 == 27:
                pos_transformed = tf2_geometry_msgs.do_transform_pose(marker.pose, trans)
                position = [pos_transformed.pose.position.x, pos_transformed.pose.position.y, pos_transformed.pose.position.z]
                if not closest_joint:
                    closest_joint = position
                else:
                    closest = np.sqrt(closest_joint[0] ** 2 + closest_joint[1] ** 2 + closest_joint[2] ** 2)
                    current_dist = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
                    if closest > current_dist:
                        closest_joint = position

        past_joint4 = self.past_pose[3]
        past_joint3 = self.past_pose[2]
        past_joint2 = self.past_pose[1]
        past_joint1 = self.past_pose[0]
        current_joint4 = self.current_pose[3]
        current_joint3 = self.current_pose[2]
        current_joint2 = self.current_pose[1]
        current_joint1 = self.current_pose[0]
        change = [np.abs(past_joint4 - current_joint4) / np.mean([past_joint4, current_joint4]),
                  np.abs(past_joint3 - current_joint3) / np.mean([past_joint3, current_joint3]),
                  np.abs(past_joint2 - current_joint2) / np.mean([past_joint2, current_joint2]),
                  np.abs(past_joint1 - current_joint1) / np.mean([past_joint1, current_joint1])
                  ]
        #only store the data when sufficient change was made by Shutter
        if change[0] < 0.1 and change[1] < 0.1 and change[2] < 0.1 and change[3] < 0.1:
            pass
        else:
        # write state and action (offset motion) to disk update past position
            self.past_pose[3] = self.current_pose[3]
            self.past_pose[2] = self.current_pose[2]
            self.past_pose[1] = self.current_pose[1]
            self.past_pose[0] = self.current_pose[0]
            if self.fid is not None:
                self.fid.write("%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" %
                            (msg.header.frame_id, closest_joint[0], closest_joint[1], closest_joint[2],
                                past_joint3, past_joint1, current_joint3, current_joint1))
                self.fid.flush()


if __name__ == '__main__':
    try:
        node = ExpertNode()
    except rospy.ROSInterruptException:
        pass