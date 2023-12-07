#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge, CvBridgeError
from shutter_lookat.msg import Target
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Bool
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

import simpleaudio as sa

def play_wav(filename):
    wave_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wave_obj.play()
    play_obj.wait_done()




class ConsentNode:
    """
    Node that asks user for consent
    """

    def __init__(self):
        """
        Constructor
        """

        self.text = "Hello! Would you like your photograph taken? (y/n)"
        # Init the node
        rospy.init_node('consent_tts')
    
        rospy.Subscriber('/new_person', Bool, self.callback)
        self.prompt_publisher = rospy.Publisher('/prompt', String, queue_size=5)
        self.record_trigger = rospy.Publisher('/record', Bool, queue_size=5)

        rospy.spin()
    
    def callback(self, new_person):
        if new_person.data:
            play_wav('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/Shutter.wav')
            # Replace with microphone and GPT API call
            print('consent played')
            self.record_trigger.publish(True)

            # prompt = input("Please provide a prompt for your image:\n")
            # self.prompt_publisher.publish(prompt)
            # Tell Tetsu's code (second stage of pipeline) to get started
            return
        

if __name__ == '__main__':
    try:
        node = ConsentNode()
    except rospy.ROSInterruptException:
        pass


