#!/usr/bin/env python3

import rospy

# to get commandline arguments
import sys

# because of transformations
import tf

import tf2_ros
import geometry_msgs.msg




if __name__ == '__main__':
    rospy.init_node('my_static_tf2_broadcaster')
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    
    static_transformStamped = geometry_msgs.msg.TransformStamped()

    static_transformStamped.header.stamp = rospy.Time.now()
    static_transformStamped.header.frame_id = "base_footprint"
    static_transformStamped.child_frame_id = "camera_base"

    static_transformStamped.transform.translation.x = -0.2
    static_transformStamped.transform.translation.y = 0.1
    static_transformStamped.transform.translation.z = 0.545

    quat = tf.transformations.quaternion_from_euler(
              0, 0, 0)
    static_transformStamped.transform.rotation.x = quat[0]
    static_transformStamped.transform.rotation.y = quat[1]
    static_transformStamped.transform.rotation.z = quat[2]
    static_transformStamped.transform.rotation.w = quat[3]
    
    
    broadcaster.sendTransform(static_transformStamped)
    rospy.spin()
