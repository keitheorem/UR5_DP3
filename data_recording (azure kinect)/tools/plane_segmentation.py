import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops

def main():
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # Set white balance
    k4a.whitebalance = 4500    
    assert k4a.whitebalance == 4500

    # Initialize the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    capture = k4a.get_capture()
    if capture is not None and capture.depth is not None and capture.color is not None:
        points = capture.depth_point_cloud.reshape((-1, 3))
        colors = capture.transformed_color[..., (2, 1, 0)].reshape((-1, 3)) / 255.0 

        # Define bounding box [min_x, min_y, min_z, max_x, max_y, max_z]
        bbox = [-500,-500,-600,1000,250,1200]
        min_bound = np.array(bbox[:3])
        max_bound = np.array(bbox[3:])  
        #crop point clouds
        indices = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        points = points[indices]
        colors = colors[indices]
        
        # Uncomment this to find plane equation to be used for plane segmentation 
        # # Create an Open3D PointCloud object
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)

        # # Perform plane segmentation
        # plane_model, inliers = pcd.segment_plane(distance_threshold=5,
        #                                         ransac_n=3,
        #                                         num_iterations=1000)
        # [a, b, c, d] = plane_model
        # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # # Separate inliers and outliers
        # inlier_cloud = pcd.select_by_index(inliers)
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])  # Paint inliers red
        # outlier_cloud = pcd.select_by_index(inliers, invert=True)

        # # Convert inlier cloud to numpy array
        # outlier_points = np.asarray(outlier_cloud.points)

        # # Now you have the inliers as a numpy array
        # print(outlier_points.shape)

        # Calculate distances of all points from the plane
        distances = distance_from_plane(points)

        # define a distance threshold
        distance_threshold = 10

        # Filter points based on the distance threshold
        indices = np.where(points[distances > distance_threshold])
        colors = colors[indices[0]]
        non_plane_points = points[distances > distance_threshold]
        
        non_plane_points = downsample_with_fps(non_plane_points)

        print(non_plane_points.shape)

        # create PointCloud object for the non-plane points
        non_plane_cloud = o3d.geometry.PointCloud()
        non_plane_cloud.points = o3d.utility.Vector3dVector(non_plane_points)
        non_plane_cloud.colors = o3d.utility.Vector3dVector(colors) 
        
        # Visualize the point clouds
        o3d.visualization.draw_geometries([non_plane_cloud],
                                          zoom=0.8,
                                          front=[-0.4999, -0.1659, -0.8499],
                                          lookat=[2.1813, 2.0619, 2.0999],
                                          up=[0.1204, -0.9852, 0.1215])

def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]

    return points

def distance_from_plane(points):
    #define the plane equation (determined from plane segementation algorithm)
    a = 0.10
    b = 0.63 
    c = 0.77
    d = -619.13 
    #calculate distance of each point from the plane (formula for shortest distance from point to plane)
    distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2) 
    return distances

if __name__ == "__main__":
    main()