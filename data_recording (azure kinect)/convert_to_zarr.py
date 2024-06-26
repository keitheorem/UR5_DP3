import zarr 
import numpy as np
import os 
import re

## Script to convert data collected in npy to zarr for model training 

save_data_path = '/home/mainuser/robot_arm/testing_code/hardware/azure_kinect/zarr_data' #path for output zarr_data
demo_data = '/home/mainuser/robot_arm/testing_code/hardware/azure_kinect/data' #path of demostration data created by DP3_Collect_Data.py 

# Open or create the Zarr root group
zarr_root = zarr.open_group(save_data_path, mode='a')  # 'a' mode opens the group for reading and writing, creating it if it doesn't exist

# Check for 'data' group
if 'data' not in zarr_root:
    zarr_data = zarr_root.create_group('data')
    print("Created 'data' group")
else:
    zarr_data = zarr_root['data']
    print("'data' group already exists")

# Check for 'meta' group
if 'meta' not in zarr_root:
    zarr_meta = zarr_root.create_group('meta')
    print("Created 'meta' group")
else:
    zarr_meta = zarr_root['meta']
    print("'meta' group already exists")

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1) 

# Regular expression to match and extract episode number
episode_pattern = re.compile(r'episode_(\d+)')

# List all items in the main folder
all_items = os.listdir(demo_data)

point_cloud_array = []
state_array = []
action_array = []
episode_end = [] 

episode_folders = sorted(
    [item for item in all_items if os.path.isdir(os.path.join(demo_data, item)) and episode_pattern.match(item)],
    key=lambda x: int(episode_pattern.match(x).group(1))
)

# Gather data from every episode into a single variable
for episode_folder in episode_folders:
    episode_path = os.path.join(demo_data, episode_folder)
    # print(f"Opening folder: {episode_path}")

    # Load the data from saved folder               
    robot_state = np.load(os.path.join(episode_path, "robot_state.npy"))
    action = np.load(os.path.join(episode_path, "action.npy"))
    point_cloud = np.load(os.path.join(episode_path, "point_cloud.npy"))

    # Append all saved data
    for element in robot_state:
        state_array.append(element)

    for element in action:
      action_array.append(element)

    for element in point_cloud:
      point_cloud_array.append(element)

    # Append when the episode ends
    if episode_end: 
      episode_end.append(episode_end[-1]+point_cloud.shape[0])
    else:
      episode_end.append(point_cloud.shape[0]) 


# Convert to numpy array
state_array = np.array(state_array, dtype = object)
action_array = np.array(action_array, dtype = object)
point_cloud_array = np.array(point_cloud_array, dtype = object)   
episode_end = np.array(episode_end, dtype = object)

# Initialise chunk Size
point_cloud_chunk_size = (1000, point_cloud_array.shape[1], point_cloud_array.shape[2])
action_chunk_size = (1000, action_array.shape[1])
state_chunk_size = (1000, state_array.shape[1])

# Create zarr datasets 
zarr_data.create_dataset('point_cloud', data=point_cloud_array, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_array, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_array, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_end, chunks=(1000,), dtype='int64', overwrite=True, compressor=compressor)