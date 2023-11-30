#!/usr/bin/env python3

import os
from datetime import datetime
import rospkg
import rospy
from sensor_msgs.msg import JointState
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import tf.transformations as tft
from scipy.optimize import least_squares
from urdf_parser_py.urdf import URDF
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray


def transform_msg_to_T(trans):
    """
    Convert TransformStamped message to 4x4 transformation matrix
    :param trans: TransformStamped message
    :return:
    """
    # extract relevant information from transform
    q = [trans.transform.rotation.x,
         trans.transform.rotation.y,
         trans.transform.rotation.z,
         trans.transform.rotation.w]
    t = [trans.transform.translation.x,
         trans.transform.translation.y,
         trans.transform.translation.z]
    # convert to matrices
    Rq = tft.quaternion_matrix(q)
    Tt = tft.translation_matrix(t)
    return np.dot(Tt, Rq)

def make_joint_rotation(angle, rotation_axis='x'):
    """
    Make rotation matrix for joint (assumes that joint angle is zero)
    :param angle: joint angle
    :param rotation_axis: rotation axis as string or vector
    :return: rotation matrix
    """
    # set axis vector if input is string
    if not isinstance(rotation_axis,list):
        assert rotation_axis in ['x', 'y', 'z'], "Invalid rotation axis '{}'".format(rotation_axis)
        if rotation_axis == 'x':
            axis = (1.0, 0.0, 0.0)
        elif rotation_axis == 'y':
            axis = (0.0, 1.0, 0.0)
        else:
            axis = (0.0, 0.0, 1.0)
    else:
        axis = rotation_axis
    # make rotation matrix
    R = tft.rotation_matrix(angle, axis)
    return R

def target_in_camera_frame(angles, target_pose, rotation_axis1, rotation_axis2, T1, T2):
    """
    Transform target to camera frame
    :param angles: joint angles
    :param target_pose: target pose
    :param rotation_axis1: str representation for the rotation axis of joint1
    :param rotation_axis2: str representation for the rotation axis of joint3
    :param T1: transform - base_link to biceps
    :param T2: transform - biceps to camera_link
    :return: target in camera_link, target in base_link
    """

    # make transform for joint 1
    R1 = make_joint_rotation(angles[0], rotation_axis=rotation_axis1)

    # make transform for joint 3
    R2 = make_joint_rotation(angles[1], rotation_axis=rotation_axis2)

    # transform target to camera_link...
    p = np.array([[target_pose[0], target_pose[1], target_pose[2], 1.0]]).transpose()

    # target in base_link
    p1 = np.dot(np.dot(T1, R1), p)

    # target in camera_link
    result = np.dot(np.dot(T2, R2), p1)

    return result[0:2].flatten()



class ExpertNode(object):
    """
    Node that simulates an expert controller using optimization. It controls two joints of the robot to make it
    point towards a target.
    """

    def __init__(self):
        rospy.init_node('expert_opt')

        # params
        self.base_link = rospy.get_param("~base_link", "base_link")
        self.biceps_link = rospy.get_param("~biceps_link", "biceps_link")
        self.camera_link = rospy.get_param("~camera_link", "camera_color_optical_frame")
        self.rgb_camera_link = rospy.get_param("~rgb_camera_link", "rgb_camera_link")
        self.human_pose_topic = rospy.get_param('~human_pose_topic', '/body_tracking_data')

        # joint values
        self.current_pose = None #[0.0, 0.0, 0.0, 0.0]
        self.joint_pub = rospy.Publisher("/joint_group_controller/command", Float64MultiArray, queue_size=5)
        # get robot model
        self.robot = URDF.from_parameter_server()

        # tf subscriber
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.nose_detected = True

        # joint subscriber
        rospy.Subscriber('/joint_states', JointState, self.joints_callback, queue_size=5)
        #subscribe to human body joints
        rospy.Subscriber(self.human_pose_topic, MarkerArray, self.body_callback, queue_size=5)

        #rospy.Subscriber('/nose_detection', Bool, self.nose_callback, queue_size=5)
        rospy.spin()

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

        self.current_pose = [msg.position[joint1_idx],
                             msg.position[joint2_idx],
                             msg.position[joint3_idx],
                             msg.position[joint4_idx]]

    def get_p_T1_T2(self, pose_transformed):
        """
        Helper function for compute_joints_position()
        :param msg: target message
        :return: target in baselink, transform from base_link to biceps, transform from biceps to camera
        """

        p = [pose_transformed[0],
             pose_transformed[1],
             pose_transformed[2]]

        # get transform from base link to camera link (base_link -> biceps_link and biceps_link -> camera_link)
        try:
            transform = self.tf_buffer.lookup_transform(self.biceps_link,
                                                        self.base_link,  # source frame
                                                        rospy.Time(0),
                                                        rospy.Duration(1.0))  # wait for 1 second
            T1 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("Failed to compute new position for the robot because {}".format(e))
            T1 = None

        try:
            transform = self.tf_buffer.lookup_transform(self.camera_link,
                                                        self.biceps_link,  # source frame
                                                        rospy.Time(0),
                                                        rospy.Duration(1.0))  # wait for 1 second
            T2 = transform_msg_to_T(transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(e)
            T2 = None

        return p, T1, T2

    def compute_joints_position(self, pose_transformed, joint1, joint3):
        """
        Helper function to compute the required motion to make the robot's camera look towards the target
        :param msg: target message
        :param joint1: current joint 1 position
        :param joint3: current joint 3 position
        :return: new joint positions for joint1 and joint3; or None if something went wrong
        """
        p, T1, T2 = self.get_p_T1_T2(pose_transformed)
        if p is None or T1 is None or T2 is None:
            return None

        # compute the required motion for the robot using black-box optimization
        x0 = [-np.arctan2(p[1], p[0]), 0.0]
        res = least_squares(target_in_camera_frame, x0,
                            bounds=([-np.pi, -np.pi * 0.5], [np.pi, np.pi * 0.5]),
                            args=(p, self.robot.joints[1].axis, self.robot.joints[3].axis, T1, T2))
        # print("result: {}, cost: {}".format(res.x, res.cost))

        offset_1 = -res.x[0]
        offset_3 = -res.x[1]

        # cap offset for joint3 based on joint limits
        if joint3 + offset_3 > self.robot.joints[3].limit.upper:
            new_offset_3 = offset_3 + self.robot.joints[3].limit.upper - (joint3 + offset_3)
            rospy.loginfo("Computed offset of {} but this led to exceeding the joint limit ({}), "
                          "so the offset was adjusted to {}".format(offset_3, self.robot.joints[3].limit.upper,
                                                                    new_offset_3))
        elif joint3 + offset_3 < self.robot.joints[3].limit.lower:
            new_offset_3 = offset_3 + self.robot.joints[3].limit.lower - (joint3 + offset_3)
            rospy.loginfo("Computed offset of {} but this led to exceeding the joint limit ({}), "
                          "so the offset was adjusted to {}".format(offset_3, self.robot.joints[3].limit.lower,
                                                                    new_offset_3))
        else:
            new_offset_3 = offset_3

        new_j1 = joint1 + offset_1
        new_j3 = joint3 + new_offset_3

        return new_j1, new_j3
    
    def body_callback(self, msg):
        """
        Target callback
        :param msg: target message
        """
        if self.current_pose is None:
            rospy.logwarn("Joint positions are unknown. Waiting to receive joint states.")
            return

        trans = self.tf_buffer.lookup_transform( 'base_footprint', 'rgb_camera_link', rospy.Time(0), rospy.Duration(1))
        closest_joint = []
        #track the closest body joint
        if self.nose_detected:
            for marker in msg.markers:
                #track the nose position
                if marker.id % 100 == 27:
                    pos_transformed = tf2_geometry_msgs.do_transform_pose(marker, trans)
                    position = [pos_transformed.pose.position.x, pos_transformed.pose.position.y, pos_transformed.pose.position.z]
                    if not closest_joint:
                        closest_joint = position
                    else:
                        closest = np.sqrt(closest_joint[0] ** 2 + closest_joint[1] ** 2 + closest_joint[2] ** 2)
                        current_dist = np.sqrt(position[0] ** 2 + position[1] ** 2 + position[2] ** 2)
                        if closest > current_dist:
                            closest_joint = position
            joint_angles = self.compute_joints_position(position, self.current_pose[0], self.current_pose[2])
            if joint_angles is None:
                # we are done. the node might not be ready yet...
                return
            else:
                # upack results
                new_j1, new_j3 = joint_angles
            print(position)
            print(joint_angles)
            msg = Float64MultiArray()
            msg.data = [float(new_j1), float(-1.5), float(new_j3), float(0.0)]
            self.joint_pub.publish(msg)
        else:
            msg = Float64MultiArray()
            msg.data = [float(0.0), float(-1.5), float(-1.5), float(0.0)]
            self.joint_pub.publish(msg)


    def nose_callback(self, msg):
        self.nose_detected = msg


if __name__ == '__main__':
    try:
        node = ExpertNode()
    except rospy.ROSInterruptException:
        pass
