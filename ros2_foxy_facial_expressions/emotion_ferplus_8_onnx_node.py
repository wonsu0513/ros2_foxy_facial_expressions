# Based on this paper;https://arxiv.org/pdf/1608.01041.pdf
# And, this github; https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
# this blog: https://bleedai.com/facial-expression-recognition-emotion-recognition-with-opencv/

# requirement for python3
#pip install bleedfacedetector
#pip install opencv-contrib-python
### Download emotion-ferplus-8.onnx from this site:
# https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus/model


### ROS2 Lib.
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
### ROS2 Msg
from std_msgs.msg import String, Int32, Float32, Int32MultiArray, Float32MultiArray
from sensor_msgs.msg import CompressedImage, Image
import sys


### OpenCV Libs
import cv2
from cv_bridge import CvBridge, CvBridgeError

### Facual libs.
import numpy as np
import matplotlib.pyplot as plt
import os
import bleedfacedetector as fd
import time

class ros2_reading_facial_expression(Node):
    def __init__(self):
        super().__init__('emotion_ferplus_8_onnx_node')
        #====================================================#
        ####  ROS2 Parameters                             ####
        #====================================================#
        self.declare_parameter('input_raw_camera', '/camera/color/image_raw')
        self.declare_parameter('input_compressed_camera', '/camera/color/image_raw/compressed')

        self.declare_parameter('compressed_input_mode', True)
        self.declare_parameter('GPU_mode', False) # True:: GPU mode / False: CPU mode


        self.param_compressed_input = self.get_parameter('compressed_input_mode').value
        self.param_gpu_mode = self.get_parameter('GPU_mode').value

        #====================================================#
        ####  Global Variables in Class                   ####
        #====================================================#
        self.bridge = CvBridge()

        self.cctv_experiment_command = 0
        self.fps = 0
        
        
        self.emotion_types = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
        # Initialize the DNN module
        self.fer_model = "./ros2_foxy_facial_expressions/emotion-ferplus-8.onnx"
        self.fer_net = cv2.dnn.readNetFromONNX(self.fer_model)
        
        if self.param_gpu_mode: # GPU mode : On
            self.fer_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.fer_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        

        #====================================================#
        ####  Subscription Section                        ####
        #====================================================#
        if self.param_compressed_input: # compressed image mode
            self.param_input_compressed_camera = self.get_parameter('input_compressed_camera').value
            self.sub_input_camera = self.create_subscription(CompressedImage, self.param_input_compressed_camera, self.sub_input_camera_callback, 10)

        else: # Raw image mode
            self.param_input_raw_camera = self.get_parameter('input_raw_camera').value
            self.sub_input_camera = self.create_subscription(Image, self.param_input_raw_camera, self.sub_input_camera_callback, 10)


        #====================================================#
        ####  Publisher Section                           ####
        #====================================================#
        # Pub:: User Study Status
        self.pub_facial_probablities= self.create_publisher(Float32MultiArray, 'behaviral_sensor/facial_expressions/probablities', 10) #The raw EEG data from the headset.
        self.pub_facial_emotion= self.create_publisher(String, 'behaviral_sensor/facial_expressions/predicted_emotion', 10) #The raw EEG data from the headset.

        
    # Sub: For reading sensors of the SMARTmBOT  
    def sub_input_camera_callback(self, msg):
        start_time = time.time()
        if self.param_compressed_input == False:
            frame = self.bridge.imgmsg_to_cv2(msg,"bgr8")
        else:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        image = cv2.flip(frame,1)
        image = self.emotion(image, returndata=True) #returndata=False --> Show the results via imshow
                
        self.fps= (1.0 / (time.time() - start_time))
        

    def emotion(self, image, returndata=False):
        # Make copy of  image
        img_copy = image.copy()
        
        # Detect faces in image
        faces = fd.ssd_detect(img_copy,conf=0.2)
        # Define padding for face ROI
        padding = 3 
        
        if not faces:
            pass

        else:
            # Iterate process for all detected faces
            for x,y,w,h in faces:          
                # Get the Face from image
                face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
                # Convert the detected face from BGR to Gray scale
                try:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                except:
                    return

                # Resize the gray scale image into 64x64
                resized_face = cv2.resize(gray, (64, 64))
                # Reshape the final image in required format of model
                processed_face = resized_face.reshape(1,1,64,64)

                # Input the processed image
                self.fer_net.setInput(processed_face)
                # Forward pass
                Output = self.fer_net.forward()
        
                # Compute softmax values for each sets of scores  
                expanded = np.exp(Output - np.max(Output))
                probablities =  expanded / expanded.sum()
                
                # Get probablityies of each emotion type
                #print('probablities===', probablities) ### Need to add array topic.
                
                # Get the final probablities by getting rid of any extra dimensions 
                prob = np.squeeze(probablities)
                #print('prob===', prob) ### Need to add array topic.

                # Get the predicted emotion            
                predicted_emotion = self.emotion_types[prob.argmax()]
                #print(predicted_emotion)  ## Need to add string topic.

                # Write predicted emotion on image
                cv2.putText(img_copy,'{}'.format(predicted_emotion),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 
                                2, cv2.LINE_AA)
                # Draw a rectangular box on the detected face
                cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
                
                # Publish results into ros2
                #### Emotion Probablities: 
                # ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
                self.pub_facial_probablities.publish(Float32MultiArray(data=prob))
                #### Emotion type: 
                # ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
                self.pub_facial_emotion.publish(String(data=predicted_emotion))

        if  returndata:
            # Return the the final image if return data is True
            return img_copy

        else:
            # Displpay the image
            cv2.imshow("Facial Expression window", img_copy)
            cv2.waitKey(3)  
       

    def sub_cctv_experiment_command_callback(self, msg):
        try:
            self.cctv_experiment_command = msg.data
        except:
            pass
        

def main(args=None):
    rclpy.init(args=args)

    facial_expression_node = ros2_reading_facial_expression()

    try:
        while rclpy.ok():
            rclpy.spin(facial_expression_node)

    except KeyboardInterrupt:
        print('repeater stopped cleanly')
        
    except BaseException:
        print('exception in repeater:', file=sys.stderr)
        raise

    finally:        
        facial_expression_node.destroy_node()
        rclpy.shutdown() 

if __name__ == '__main__':
    main()




