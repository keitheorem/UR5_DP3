import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import keyboard
import threading
import robotiq_gripper
from pynput import keyboard 
import zarr
import csv
from rtde_receive import RTDEReceiveInterface 
import shutil
import torch
import pytorch3d.ops as torch3d_ops

## Script to collect demostrations of task for robot 

## Press 'c' to start recording new episode 
## Press 's' to stop recording 
## Press 'd' to delete most recent episode 
## Press 'q' to quit 

class AppState:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
    
#Declarations 
state = AppState()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
decimate = rs.decimation_filter()
decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
colorizer = rs.colorizer()

# Define robot parameters
ROBOT_HOST = "192.168.20.124"  # IP address of the robot controller
print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ROBOT_HOST, 63352)
print("Activating gripper...")
gripper.activate()

# UR5 RTDE receiver setup
rtde_r = RTDEReceiveInterface(ROBOT_HOST)

# Global variables for controlling the recording state
data_folder = "data"
recording = False
current_episode = 0

# Function to get robot state
def get_robot_state():
    state = np.array(rtde_r.getActualTCPPose())
    action = np.array(rtde_r.getTargetTCPPose())
    gripper_state = np.array([gripper.get_current_position()]) 
    state = np.concatenate((state,gripper_state))
    action = np.concatenate((action,gripper_state))
    return state, action

def get_point_clouds():
  # Wait for a coherent pair of frames: depth and color
  frames = pipeline.wait_for_frames()

  depth_frame = frames.get_depth_frame()
  color_frame = frames.get_color_frame()

  depth_frame = decimate.process(depth_frame)

  # Grab new intrinsics (may be changed by decimation)
  depth_intrinsics = rs.video_stream_profile(
      depth_frame.profile).get_intrinsics()
  w, h = depth_intrinsics.width, depth_intrinsics.height

  depth_image = np.asanyarray(depth_frame.get_data())
  color_image = np.asanyarray(color_frame.get_data())

  depth_colormap = np.asanyarray(
      colorizer.colorize(depth_frame).get_data())

  if state.color:
      mapped_frame, color_source = color_frame, color_image
  else:
      mapped_frame, color_source = depth_frame, depth_colormap

  points = pc.calculate(depth_frame)
  pc.map_to(mapped_frame)

  # Pointcloud data to arrays
  v, t = points.get_vertices(), points.get_texture_coordinates()
  verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz

  # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
  bbox = [-0.4, -0.5, -2, 0.37, 0.7, 2]
  min_bound = np.array(bbox[:3])
  max_bound = np.array(bbox[3:])  
  #crop point clouds
  indices = np.all((verts >= min_bound) & (verts <= max_bound), axis=1)
  verts = verts[indices]

  return verts

def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

# Function to handle key presses
def on_press(key):
    global recording, current_episode

    try:
        if key.char == 'c':
            if not recording:
                current_episode += 1
                recording = True
                print(f"Started recording episode {current_episode}...")

        elif key.char == 's':
            if recording:
                recording = False
                episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                print(f"Stopped recording episode {current_episode}")

                #reload and reshape data
                time.sleep(1)
                reshape_data(episode_folder)
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
def synchronized_capture(frequency=30):
    global recording, current_episode

    interval = 1.0 / frequency

    while True:
        start_time = time.time()

        if recording:
            # Get robot state
            robot_state, action = get_robot_state()

            # Get point_cloud
            point_cloud = get_point_clouds()
            point_cloud = downsample_with_fps(np.array(point_cloud))

            if robot_state is not None:
                # Create episode folder if it doesn't exist
                episode_folder = os.path.join(data_folder, f"episode_{current_episode}")
                if not os.path.exists(episode_folder):
                    os.makedirs(episode_folder)

                # Save data
                save_data(episode_folder, robot_state, action, point_cloud)
                print("loading")

        elapsed_time = time.time() - start_time
        time_to_sleep = interval - elapsed_time
        

        if time_to_sleep > 0:
            time.sleep(time_to_sleep)


def save_data(episode_folder, robot_state, action, point_cloud):
    # Check if the episode folder exists, create it if not
    if not os.path.exists(episode_folder):
        os.makedirs(episode_folder)

    # Save or append data to npy files
    robot_state_file = os.path.join(episode_folder, "robot_state.npy")
    action_file = os.path.join(episode_folder, "action.npy")
    point_cloud_file = os.path.join(episode_folder, "point_cloud.npy")

    if os.path.exists(robot_state_file):
        existing_robot_state = np.load(robot_state_file)
        robot_state = np.concatenate((existing_robot_state, robot_state), axis=0)
    np.save(robot_state_file, robot_state)

    if os.path.exists(action_file):
        existing_action = np.load(action_file)
        action = np.concatenate((existing_action, action), axis=0)
    np.save(action_file, action)

    if os.path.exists(point_cloud_file):
        existing_point_cloud = np.load(point_cloud_file)
        point_cloud = np.concatenate((existing_point_cloud, point_cloud), axis=0)
    np.save(point_cloud_file, point_cloud)

def reshape_data(episode_folder, num_points = 1024, action_shape=7, state_shape=7):
    #Define file paths 
    robot_state_file = os.path.join(episode_folder, "robot_state.npy")
    action_file = os.path.join(episode_folder, "action.npy")
    point_cloud_file = os.path.join(episode_folder, "point_cloud.npy")

    #Load file path
    robot_state = np.load(robot_state_file)
    action = np.load(action_file)
    point_cloud = np.load(point_cloud_file)

    #Calculate frames 
    T = point_cloud.shape[0] // num_points

    #Reshape point clouds and save
    point_cloud = point_cloud.reshape(T,num_points,3)
    np.save(point_cloud_file, point_cloud)

    #Reshape states and save
    robot_state = robot_state.reshape(T,state_shape)
    np.save(robot_state_file, robot_state)

    #Reshape actions and save
    action = action.reshape(T,action_shape)
    np.save(action_file, action)

def main():
    global current_episode

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
    capture_thread = threading.Thread(target=synchronized_capture)
    capture_thread.start()

    # Start listening for keyboard inputs
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    listener.join()

    print("Stopped synchronized capture.")

if __name__ == '__main__':
    main()