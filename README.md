remote: Support for password authentication was removed on August 13, 2021.
remote: Please see https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls for information on currently recommended modes of authentication.
fatal: Authentication failed for 'https://github.com/yuzhench/rosbag_process.git/'# rosbag_process

1. after you have the .bag file, first pass the bag2csv.py program:

     ---> it will check how many .bag file in your folder and automatically generate a new folder with the same name as the .bag file.
   
     ---> inside of the new folder, there are two subfolder called depth and rgb.

   
2. use the rgbd_process to generate
     ---> it will create a result folder
   
     ---> the final result will in the result_out subfolder

3. after getting the .npy files, you can then pass the .npy files (both depth and semantic) to kinfu

   ---> you will run the kinfu and generate the kindu result video.
   
   ---> when you run the kinfu, remember to modify the mian() function so that it will load each frame to a folder instead of showing the result video in the screen 
   
   
4. if the kinfu show the error message about the data type of the .npy file, try to run the uint8_to_int64.py to change the data type 



5. once you get the frames, you can put them in the same foler and share in the channel






rosbag recording steps

window1: 
1. source devel/setup.bash
2. roscore


window2:
1. source devel/setup.bash
2. roslaunch src/Azure_Kinect_ROS_Driver/launch/driver.launch


window3:
1. source devel/setup.bash
2. rosbag record (the necessary topics)

topics
/clicked_point
/clock
/depth_to_rgb/camera_info
/depth_to_rgb/image_raw
/initialpose
/move_base_simple/goal
/rgb/camera_info
/rgb/image_raw
/rosout
/rosout_agg
/tf
/tf_static
   
