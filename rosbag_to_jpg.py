
import cv2
import numpy as np
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

bag_file = '/home/yuzhen/SegFormer-tensorrt/rosbag2.bag'  # Replace with the path to your ROS bag file
output_directory = '/home/yuzhen/SegFormer-tensorrt/rosbag_output_rgb'  # Replace with the desired output directory

# Open the ROS bag file
bag = rosbag.Bag(bag_file, 'r')

# Create a CvBridge object
bridge = CvBridge()




# clean the directory before we input the new files
import os

def clean_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

# Clean the directory
clean_directory(output_directory)










count = 0
# Iterate over the messages in the bag
for topic, msg, t in bag.read_messages(topics=['/rgb/image_raw']):
    # Check if the message is of type sensor_msgs/Image (RGB image)
    if msg._type == 'sensor_msgs/Image':
        # Extract the image data from the message
        img_data = np.frombuffer(msg.data, dtype=np.uint8)

        # Reshape the image data into a 2D array
        img = img_data.reshape((msg.height, msg.width, -1))
        # print(f"{msg.height} , {msg.width}")
        # break
        # Convert the image from BGR to RGB format
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save the image as JPEG
        image_path = f'{output_directory}/{count}_semantic.jpg'
        cv2.imwrite(image_path, img)
        count+=1

# Close the bag file
bag.close()

