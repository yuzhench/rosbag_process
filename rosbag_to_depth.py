import rosbag
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bag_file = '/home/yuzhen/SegFormer-tensorrt/For_KINFU_mapping_without_tape.bag'  # Replace with the path to your ROS bag file
output_dir =  "/home/yuzhen/Desktop/first/Azure-Kinect-Samples/opencv-kinfu-samples/build/For_KINFU_mapping_without_tape_out/individual_depth_files"  # Replace with the directory where you want to save the JPG images
# Open the ROS bag file
bag = rosbag.Bag(bag_file)

# Initialize the CvBridge
bridge = CvBridge()





# clean the directory before we input the new files
import os

def clean_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            file_path = os.path.join(directory, filename)
            os.remove(file_path)

# Clean the directory
clean_directory(output_dir)





count = 0
# Iterate over the messages in the bag file
for topic, msg, t in bag.read_messages(topics=['/depth_to_rgb/image']):
    #print("enter the loop")
    # Convert the ROS sensor_msgs/Image message to a NumPy array
    
    depth_data = np.frombuffer(msg.data, dtype=np.uint8) #.reshape(msg.height, msg.width,-1)
    #print(f"{msg.height} , {msg.width}")

    # Convert the depth data to grayscale
    depth_data_gray = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)


    # Create a grayscale image from the depth data
    plt.imshow(depth_data_gray, cmap='gray')
    plt.axis('off')

    # Save the grayscale image as a JPG file
    output_file = f"{output_dir}/{count}_depth.jpg"  # Use the timestamp as the filename
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
    plt.close()
    count+=1
# Close the ROS bag file
bag.close()













# import rosbag
# import cv2
# from cv_bridge import CvBridge
# import os

# # Specify the path to your ROS bag file
# bag_file_path = "/home/yuzhen/SegFormer-tensorrt/rosbag1.bag"

# # Specify the topic containing the depth image
# topic_name = "/depth_to_rgb/image"

# # Specify the output directory for saving the JPG files
# output_directory = "/home/yuzhen/SegFormer-tensorrt/rosbag_output_depth_image"

# # Create the output directory if it doesn't exist
# if not os.path.exists(output_directory):
#     os.makedirs(output_directory)

# # Open the ROS bag file
# bag = rosbag.Bag(bag_file_path)

# # Initialize cv_bridge
# cv_bridge = CvBridge()

# # Initialize a counter for the saved images
# image_count = 1

# # Iterate over the messages in the bag file
# for topic, msg, t in bag.read_messages(topics=[topic_name]):
#     # Convert the depth image message to a OpenCV image
#     cv_image = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

#     # Save the image as a JPG file
#     image_filename = f"depth_image_{image_count}.jpg"
#     image_path = os.path.join(output_directory, image_filename)
#     cv2.imwrite(image_path, cv_image)

#     # Increment the counter
#     image_count += 1

# # Close the ROS bag file
# bag.close()
