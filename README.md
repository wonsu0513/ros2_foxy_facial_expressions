# ros2_foxy_facial_expressions
ros2_foxy_facial_expressions

(Draft)   I will update it.


This repository is made by below materials:
paper;https://arxiv.org/pdf/1608.01041.pdf
github; https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
blog: https://bleedai.com/facial-expression-recognition-emotion-recognition-with-opencv/


In order to run this nodes, you need to download the liberies for python.

# requirement for python3
#pip install bleedfacedetector
#pip install opencv-contrib-python
### Download emotion-ferplus-8.onnx from this site:
# https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus/model


You need to change parameters of input camera topics; such as

'input_raw_camera': '/camera/color/image_raw', 
'input_compressed_camera': '/camera/color/image_raw/compressed',
'compressed_input_mode': False,
'PU_mode': True,
            
            
For running this node, I recommend you to use launch file as belows;
            
facial_expression_compressed_cpu.launch.py
facial_expression_compressed_gpu.launch.py
facial_expression_raw_img_cpu.launch.py
facial_expression_raw_img_gpu.launch.py
