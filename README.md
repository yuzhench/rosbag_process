remote: Support for password authentication was removed on August 13, 2021.
remote: Please see https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls for information on currently recommended modes of authentication.
fatal: Authentication failed for 'https://github.com/yuzhench/rosbag_process.git/'# rosbag_process

1. after you have the .bag file, first pass the bag2csv.py program:

     ---> it will check how many .bag file in your folder and automatically generate a new folder with the same name as the .bag file.
   
     ---> inside of the new folder, there are two subfolder called depth and rgb.

   
3. use the rgbd_process to generate
     ---> it will create a result folder
   
     ---> the final result will in the result_out subfolder

5. after getting the .npy file, you can then pass the .npy files (both depth and semantic) to kinfu

   ---> you will run the kinfu and generate the kindu result video.
