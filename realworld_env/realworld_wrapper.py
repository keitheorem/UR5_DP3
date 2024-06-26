import gym
import numpy as np
import time
import math
import cv2
import pyk4a
from pyk4a import Config, PyK4A
from diffusion_policy_3d.env.realworld import robotiq_gripper
from gym import spaces
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
import torch
import pytorch3d.ops as torch3d_ops
import open3d as o3d
    

def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points

class RealWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 robot_ip="192.168.20.124"):
        super(RealWorldEnv, self).__init__()

        self.image_size = 128
        self.use_point_crop = use_point_crop
        self.num_points = num_points
        self.robot_ip = robot_ip

        # Initialize camera and point cloud sensors
        self.k4a = self.initialize_camera()

        # Initialize robot and gripper
        self.rtde_r, self.rtde_c, self.gripper = self.initialize_robot()

        self.episode_length = self._max_episode_steps = 200
        self.obs_sensor_dim = 7  # 6 DOF + 1 Gripper
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_sensor_dim,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
        })

    def initialize_camera(self):
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1440P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only=False,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500

        return k4a

    def initialize_robot(self):
        # Initialize the RTDE interfaces for the robot
        rtde_r = RTDEReceiveInterface(self.robot_ip)
        rtde_c = RTDEControlInterface(self.robot_ip)

        # Initialize and activate the gripper
        gripper = robotiq_gripper.RobotiqGripper()
        gripper.connect(self.robot_ip, 63352)
        gripper.activate()

        return rtde_r, rtde_c, gripper
    
    def distance_from_plane(self, points):
        #define the plane equation (determined from plane segementation algorithm)
        a = 0.10
        b = 0.63 
        c = 0.77
        d = -619.13 
        #calculate distance of each point from the plane (formula for shortest distance from point to plane)
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2) 
        return distances

    def get_point_cloud(self, use_rgb = False):
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

            distances = self.distance_from_plane(points)

            # Define a distance threshold
            distance_threshold = 10

            # Filter points based on the distance threshold
            points = points[distances > distance_threshold]
            points = downsample_with_fps(np.array(points))

        return points 
    
    def get_robot_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    def get_visual_obs(self):
        point_cloud = self.get_point_cloud()
        state, action = self.get_robot_state()
        
        # Create the observation dictionary
        obs_dict = {
            'agent_pos': state,
            'point_cloud': point_cloud,
        }

        return obs_dict

    def step(self, action):
        # Send action to the robot
        pose = action[:6]    
        gripper_position = action[6] + 3

        self.rtde_c.moveJ(pose) 
        if gripper_position < self.gripper.get_min_position():
            gripper_position = self.gripper.get_min_position()

        if gripper_position > self.gripper.get_max_position():
            gripper_position = self.gripper.get_max_position()

        gripper_position = int(gripper_position)

        self.gripper.move(gripper_position, 155, 255)
        
        obs = self.get_visual_obs()

        if self.rtde_r.getRobotMode() == 7:
            done = False
        
        else:
            done = True
    
        return obs, done

    
    def render(self, mode = 'rgb_array'):
        capture = self.k4a.get_capture()
        if capture is not None and capture.depth is not None and capture.color is not None:
            points = capture.depth_point_cloud.reshape((-1, 3))
            colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3))
            bbox = [-500,-500,-600,1000,250,1200]
            min_bound = np.array(bbox[:3])
            max_bound = np.array(bbox[3:])  
            #crop point clouds
            indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
            points = points[indices]
            colors = colors[indices]

        return colors

    def reset(self):
        # Reset the robot to the initial position
        #self.rtde_c.moveL([0, 0, 0, 0, 0, 0])
        #self.gripper.move(0, 155, 255)  # Open the gripper
        return self.get_visual_obs()


# def main(): 
#     env = RealWorldEnv('realworld_UR5')
#     pc = env.get_point_cloud()
#     non_plane_cloud = o3d.geometry.PointCloud()
#     non_plane_cloud.points = o3d.utility.Vector3dVector(pc)
#     # Visualize the point clouds
#     o3d.visualization.draw_geometries([non_plane_cloud],
#                                           zoom=0.8,
#                                           front=[-0.4999, -0.1659, -0.8499],
#                                           lookat=[2.1813, 2.0619, 2.0999],
#                                           up=[0.1204, -0.9852, 0.1215])

    
# if __name__ == '__main__':
#     main()

