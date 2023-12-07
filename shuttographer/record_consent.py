#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool  # Or whatever message type your trigger uses
from audio_common_msgs.msg import AudioData
from transformers import pipeline
import torch
import requests
import json
from consent import play_wav
import simpleaudio as sa
import helpers


class AudioRecorder:
    """
    records audio from audio_capture
    """

    def __init__(self):
        """
        Constructor
        """

        self.audio_frames = []
        self.recorded_already = False
        
        rospy.init_node('audio_recording_node', anonymous=True)
        rospy.Subscriber('/record', Bool, self.record_callback)
        self.consent_confirmed_publisher = rospy.Publisher('/consent_confirmed', Bool, queue_size=5)
        
        rospy.spin()
    
    def record_callback(self, msg):
        if msg.data and not self.recorded_already:
            self.recorded_already = True
            self.audio_subscriber = rospy.Subscriber('/audio/audio', AudioData, self.callback)


    def callback(self, msg):
        print('recording started')
        self.audio_frames.append(msg.data)
        if len(self.audio_frames) == 40:
            print('recording finished')
            with open('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/consent_output.mp3', 'wb') as mp3_file:
                for frame in self.audio_frames:
                    mp3_file.write(frame)
            self.audio_subscriber.unregister()
            text = helpers.speech_to_text('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/consent_output.mp3')
            print(f"consent given: {text}")
            consent = helpers.call_chatbot_api(text)
            print(f"chatbot response: {consent}")
            if consent == 'yes':
                play_wav('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/shutter-prompt.wav')
                self.consent_confirmed_publisher.publish(True)
                # flag for step 2 of pipeline to start
            else:
                play_wav('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/Shutter-sad.wav')
            


            
            return
        
if __name__ == '__main__':
    try:
        node = AudioRecorder()
    except rospy.ROSInterruptException:
        pass


