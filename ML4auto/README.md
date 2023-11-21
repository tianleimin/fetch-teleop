# Example scripts for supervised-learning approach to develop auto handover model using the FACT dataset

- csv_header_explained.txt: details on the columns in the extracted csv data (FACT-processed)
- ROS_bag_info.txt: details on the type of data saved in the ROS bags (FACT-raw)
- rosbag2images.py: if you want to extract raw frames from the Fetch and OAK-D cameras from the ROS bags (FACT-raw) as training data instead of using the extracted csv files (warning: this will save **a lot** of images per bag)
- Preprocess.ipynb: for preprocessing the extracted csv data, counting interesting stats, and computing example reward values based on emotion and episode handover quality annotation
- KeyPointsKeyTimes.ipynb: for extracting data segments at key transitions of the robot's state machine (start of episode, start of arm reaching at participant's side, start of arm tucking at participant's side)
- CNN2D_timing_example.ipynb: example binary classification of handover timing
- CNN2D_OTP_example.ipynb: example 3-way classification of handover location
- MLinfer_local.ipynb: prediction loop to achieve full auto handover by replaying action primitives (ROS integration part removed, prints which primitive to be executed instead of actually executing it on the robot)