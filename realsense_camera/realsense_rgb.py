import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Create a directory to save RGB images
output_dir = "rgb_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Configure RGB stream
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    frame_count = 0
    while True:
        # Wait for a coherent pair of frames: color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Save the RGB image
        rgb_image_path = os.path.join(output_dir, f"rgb_image_{frame_count:04d}.png")
        cv2.imwrite(rgb_image_path, color_image)

        # Display the RGB image
        cv2.imshow('RealSense RGB Stream', color_image)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
