import numpy as np
import cv2

class PointCloudGenerator:
    def __init__(self, focal_length, sensor_width_mm, known_distance=None):
        self.focal_length = focal_length
        self.sensor_width_mm = sensor_width_mm
        self.known_distance = known_distance

    def generate_point_cloud(self, depth_map, image):
        """Generate a point cloud from the depth map with proper scaling."""
        height, width = depth_map.shape
        scale_factor = (self.sensor_width_mm / 1000) / width  # Convert to meters

        # Create coordinate grid
        x, y = np.meshgrid(
            np.linspace(-width / 2, width / 2, width),
            np.linspace(-height / 2, height / 2, height)
        )
        x = x * scale_factor
        y = y * scale_factor

        # Scale depth to real-world units
        if self.known_distance is not None:
            depth_map *= (self.known_distance / depth_map.max())
        else:
            depth_map *= 10.0  # Default scaling factor

        # Stack coordinates
        points = np.stack([x, y, depth_map], axis=-1).reshape(-1, 3)
        valid_mask = ~np.isnan(points).any(axis=1)
        return points[valid_mask]