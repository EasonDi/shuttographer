#!/usr/bin/env python3
import rospy
from helpers import stable_diffusion_edit
import cv2
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray
import os
import portrait_evaluation.inference
from consent import play_wav

N = 0
IMG_DIRECTORY = '/home/jr2683/catkin_ws/src/shuttographer/shuttographer/img_folder'




def empty_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        os.unlink(file_path)


class PhotoNode:
    """
    Node that processes and edits photos
    """

    def __init__(self):
        """
        Constructor
        """
        
        # Init the node
        rospy.init_node('photo_node')
        self.prompt_received = False
        self.prompt = "A wizard with a wand against a starry background"
        self.scores = []
        rospy.Subscriber('/prompt', String, self.prompt_processing_callback)
        self.img_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_processing_callback)
        
        rospy.spin()

    def prompt_processing_callback(self, prompt):
        self.prompt_received = True
        self.prompt = prompt

    def image_processing_callback(self, image):
        global N
        if self.prompt_received == True:
            if N == 0:
                empty_directory(IMG_DIRECTORY)
            if  N % 40 == 0:
                bridge = CvBridge()
                try:
                    cv2_img = bridge.imgmsg_to_cv2(image, "bgr8")
                except CvBridgeError as e:
                    print(e)
                else:
                    score = portrait_evaluation.inference.get_quality_score(cv2_img)
                    self.scores.append(score)
                    cv2.imwrite(f'/home/jr2683/catkin_ws/src/shuttographer/shuttographer/img_folder/camera_image_{N}.jpg', cv2_img)
                    print (f'img {N} processed')
                    print(self.scores)    
            if N == 360:
                self.img_sub.unregister()
                self.process_photos(IMG_DIRECTORY)
            N += 1
                
    
    def process_photos(self, directory):
        max_score = max(self.scores)
        max_index = self.scores.index(max_score) * 40
        max_score_path = f'{directory}/camera_image_{max_index}.jpg'
        image = cv2.imread(max_score_path)
        resized_image = cv2.resize(image, (1216, 832))
        cv2.imwrite(max_score_path, resized_image)
        stable_diffusion_edit(self.prompt, max_score_path)
        play_wav('/home/jr2683/catkin_ws/src/shuttographer/shuttographer/audio_files/Shutter-show-photo.wav')



        
            
        
if __name__ == '__main__':
    try:
        node = PhotoNode()
    except rospy.ROSInterruptException:
        pass

