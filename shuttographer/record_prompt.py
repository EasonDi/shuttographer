#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool
from std_msgs.msg import String
from audio_common_msgs.msg import AudioData
from consent import play_wav
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
        
        rospy.init_node('prompt_recording_node', anonymous=True)
        rospy.Subscriber('/consent_confirmed', Bool, self.record_callback)
        self.prompt_publisher = rospy.Publisher('/prompt', String, queue_size=5)
        
        rospy.spin()
    
    def record_callback(self, msg):
        if msg.data and not self.recorded_already:
            self.recorded_already = True
            self.audio_subscriber = rospy.Subscriber('/audio/audio', AudioData, self.callback)


    def callback(self, msg):
        self.audio_frames.append(msg.data)
        if len(self.audio_frames) == 100:
            with open('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/prompt_output.mp3', 'wb') as mp3_file:
                for frame in self.audio_frames:
                    mp3_file.write(frame)
            self.audio_subscriber.unregister()
            text = helpers.speech_to_text('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/prompt_output.mp3')
            print(text)
            self.prompt_publisher.publish(text['text'])
            play_wav('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/Shutter-enthusiastic.wav')
            return
        
if __name__ == '__main__':
    try:
        node = AudioRecorder()
    except rospy.ROSInterruptException:
        pass


