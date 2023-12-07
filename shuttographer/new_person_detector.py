#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray


class NewPersonDetectorNode:
    """
    Node that checks whether a new person has entered kinect stream
    """

    def __init__(self):
        """
        Constructor
        """

        human_pose_topic = rospy.get_param('~human_pose_topic', '/body_tracking_data')
        
        # Init the node
        rospy.init_node('new_person_detection')
        self.first_person = True
        self.last_received = rospy.Time.now()
    
        self.body_detected = rospy.Publisher('/new_person', Bool, queue_size=5)
    
        rospy.Subscriber(human_pose_topic, MarkerArray, self.callback)
        
        rospy.spin()

    def callback(self, human_pose_topic):
            if human_pose_topic.markers == []:
                self.body_detected.publish(False)
                return
            if self.first_person:
                 self.first_person = False
                 self.last_received = rospy.Time.now()
                 self.body_detected.publish(True)
                 return
            if rospy.Time.now() - self.last_received > rospy.Duration(3):
                 self.last_received = rospy.Time.now()
                 self.body_detected.publish(True)
                 return
            self.last_received = rospy.Time.now()
            self.body_detected.publish(False)
        
        
if __name__ == '__main__':
    try:
        node = NewPersonDetectorNode()
    except rospy.ROSInterruptException:
        pass

