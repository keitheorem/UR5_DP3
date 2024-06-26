import time
import numpy as np
import pyrealsense2 as rs
import os
import threading
import robotiq_gripper
from pynput import keyboard 
from rtde_receive import RTDEReceiveInterface 
import shutil
import torch
import pytorch3d.ops as torch3d_ops
import pyk4a
from pyk4a import Config, PyK4A
import sys
import open3d as o3d
import cv2

## Script to collect demostrations of task for robot 

## Press 'c' to start recording new episode 
## Press 's' to stop recording 
## Press 'd' to delete most recent episode 
## Press 'q' to quit 

class Data():
    def __init__(self): 
        self.ROBOT_HOST = "192.168.20.124" 
        self.num_points = 1024 # Adjust number of points to downsample to here
        self.state_shape = 7 # Adjust state shape here
        self.action_shape = 7 # Adjust action state here

        #start robot and camera to receive data
        self.rtde_r, self.gripper = self.start_robot()
        self.k4a = self.start_camera()

    def start_camera(self): 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only= False,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500

        return k4a
    
    def start_robot(self): 
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        print("Creating gripper...")
        gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        gripper.connect(self.ROBOT_HOST, 63352)
        print("Activating gripper...")
        gripper.activate()
        
        return rtde_r, gripper

    # Function to get robot state
    def get_robot_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state,gripper_state))
        action = np.concatenate((action,gripper_state))

        return state, action

    def get_visual_obs(self):
      # Wait for a coherent pair of frames: depth and color
      capture = self.k4a.get_capture()
      if capture is not None and capture.depth is not None and capture.color is not None:
        points = capture.depth_point_cloud.reshape((-1, 3))
        #colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

        # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        bbox = [-500,-500,-600,1000,250,1200]
        min_bound = np.array(bbox[:3])
        max_bound = np.array(bbox[3:])  

        #crop point clouds
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        points = points[indices]
        #colors = colors[indices]

        return points
  
    def downsample_with_fps(self, points: np.ndarray):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        self.num_points = torch.tensor([self.num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=self.num_points)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points
    
    def distance_from_plane(self, points):
        #define the plane equation (determined from plane segementation algorithm)
        a = 0.10
        b = 0.63 
        c = 0.77
        d = -619.13 
        #calculate distance of each point from the plane 
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)
        return distances

    # Function to handle key presses
    def on_press(self, key):
        global recording, current_episode

        try:
            if key.char == 'c':
                if not recording:
                    current_episode += 1
                    global robot_state_array, action_array, point_cloud_array, rgb_array, depth_array
                    robot_state_array, action_array, point_cloud_array, rgb_array, depth_array = [], [], [], [], []
                    recording = True
                    print(f"Started recording episode {current_episode}...")

            elif key.char == 's':
                if recording:
                    recording = False
                    episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                    print(f"Stopped recording episode {current_episode}")

                    #reload and reshape data
                    time.sleep(1)
                    self.reshape_data(episode_folder)
                    print("Data saved")

            elif key.char == 'q':
                if recording:
                    recording = False
                print("Quitting session...")
                return False  # Stop listener
            
            elif key.char == 'd':
                if not recording:
                    if current_episode > 0:
                        confirmation = input("Are you sure you want to delete the most recent episode? (y/n): ")
                        if confirmation.lower() == 'y':
                            # Delete the episode folder
                            episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                            if os.path.exists(episode_folder):
                                shutil.rmtree(episode_folder)
                            current_episode -= 1
                            print(f"Deleted episode {current_episode}")
                        else:
                            print("Deletion canceled.")
                    else:
                        print("No episodes to delete.")


        except AttributeError:
            pass


    # Synchronization mechanism
    def synchronized_capture(self, frequency=10):
        global recording, current_episode

        interval = 1.0 / frequency

        while True:
            start_time = time.time()

            if recording:
                # Get robot state
                robot_state, action = self.get_robot_state()

                # Get point_cloud
                point_cloud = self.get_visual_obs()
                
                # Apply plane segmentation to remove tabyy
                distances = self.distance_from_plane(point_cloud)
                distance_threshold = 10 # Adjust distance threshold for plane segementation here
                point_cloud = point_cloud[distances > distance_threshold]
                
                # Downsample with FPS
                point_cloud = self.downsample_with_fps(np.array(point_cloud))

                if robot_state is not None:
                    # Create episode folder if it doesn't exist
                    episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                    if not os.path.exists(episode_folder):
                        os.makedirs(episode_folder)

                    # Save data
                    self.save_data(episode_folder, robot_state, action, point_cloud)
                    print("loading")

            elapsed_time = time.time() - start_time
            time_to_sleep = interval - elapsed_time

            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

    def save_data(self, episode_folder, robot_state, action, point_cloud):
        # Check if the episode folder exists, create it if not
        if not os.path.exists(episode_folder):
            os.makedirs(episode_folder)

        robot_state_array.append(robot_state)
        action_array.append(action)
        point_cloud_array.append(point_cloud)

    def reshape_data(self, episode_folder):
        #Define file paths 
        robot_state_file = os.path.join(episode_folder, "robot_state.npy")
        action_file = os.path.join(episode_folder, "action.npy")
        point_cloud_file = os.path.join(episode_folder, "point_cloud.npy")

        robot_state = np.array(robot_state_array)
        action = np.array(action_array)
        point_cloud = np.array(point_cloud_array)

        #Save all data into npy format
        np.save(point_cloud_file, point_cloud)
        np.save(robot_state_file, robot_state)
        np.save(action_file, action)


def main():

    global current_episode, data_folder, recording
    # Global variables for controlling the recording state
    data_folder = "data"
    recording = False
    current_episode = 0

    data = Data()

    # Check for the latest episode and continue from it
    # Ensure data folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    episodes = [int(folder.split('_')[-1]) for folder in os.listdir(data_folder) if folder.startswith('episode')]
    if episodes:
        current_episode = max(episodes)
        print(f"Resuming from episode {current_episode}")
    else:
        current_episode = 0
        print("No previous episodes found. Starting from episode 1.")

    print("Start Recording, Press C to start recording episode")

    # Start the synchronized capture in a separate thread
    capture_thread = threading.Thread(target=data.synchronized_capture)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=data.on_press)
    listener.start()
    listener.join()

    print("Stopped synchronized capture.")

if __name__ == '__main__':
    main()