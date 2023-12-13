# CPSC-459/559 Shuttographer -Imitation Learning- README

## Dependencies

[Tensorflow](https://github.com/tensorflow/tensorflow)

[Keras](https://github.com/keras-team/keras)

[scikit-learn](https://github.com/scikit-learn/scikit-learn)

## How to run the code 

[Data Collection]

- Run Kinect body tracker (Main Driver dependencies need to be installed)
     ```bash
    $ roslaunch azure_kinect_ros_driver driver_with_bodytracking.launch
    ```

- Run Shutter teleoperation
     ```bash
    $ roslaunch shutter_teleop shutter_controller.launch simulation:=false
    ```

- Run Shutter realsense2 camera
     ```bash
    $ roslaunch realsense2_camera rs_camera.launch
    ```

- To collect joint state data into an output file "joint_states.txt" while demonstration

    ```bash
    $ rostopic echo /joint_states >> joint_states.txt
    ```

- To collect Kinect's body tracking data into an output file "body_tracking_data.txt" while demonstration

    ```bash
    $ rostopic echo /body_tracking_data >> body_tracking_data.txt
    ```

[Data Processing]
- To convert the txt files obtained in the [Data Collection] phase into a state action pair dictionary
    ```bash
    $ python3 construct_state_action_pair.py <joint positions txt file> <body tracking txt file> <output json file>
    ```

[Train model and save]
- To train a deep neural network model and save it (change the file paths in behavior_cloning.py to appropriate paths when running)

    ```bash
    $ python3 behavior_cloning.py
    ```