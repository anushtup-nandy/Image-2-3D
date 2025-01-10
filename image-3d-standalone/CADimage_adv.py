import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import pyvista as pv
import open3d as o3d  # For advanced mesh generation
import gc  # Garbage collector

class AdvancedImageTo3D:
    def __init__(self, max_image_size=384, focal_length=1000, sensor_width_mm=36.0, known_distance=None):
        """Initialize with advanced settings
        Args:
            max_image_size: Maximum size for image's longest side
            focal_length: Focal length of the camera (in pixels)
            sensor_width_mm: Physical width of the camera sensor (in millimeters)
            known_distance: Known distance of an object in the scene (in meters)
        """
        self.max_image_size = max_image_size
        self.focal_length = focal_length
        self.sensor_width_mm = sensor_width_mm
        self.known_distance = known_distance  # For depth calibration
        
        # Initialize depth model with memory efficiency options
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            "Intel/dpt-hybrid-midas",  # Using smaller hybrid model instead of large
            torch_dtype=torch.float16  # Use half precision
        )
        
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas",  # Using smaller hybrid model
            torch_dtype=torch.float16
        )
        
        # Use CUDA if available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear CUDA cache
            
        self.depth_model.to(self.device)

    def resize_image(self, image):
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        scale = min(self.max_image_size / max(width, height), 1.0)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

    def load_image(self, image_path):
        """Load and preprocess the image"""
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image if too large
        image = self.resize_image(image)
        return image

    def estimate_depth(self, image):
        """Estimate depth map with memory efficiency"""
        try:
            # Convert image to tensor with memory pinning
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device, torch.float16) for k, v in inputs.items()}
            
            # Free some memory
            torch.cuda.empty_cache() if self.device.type == 'cuda' else gc.collect()
            
            # Get depth prediction
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # Move to CPU and convert to numpy
            depth_map = predicted_depth.cpu().float().numpy().squeeze()
            
            # Clear memory
            del inputs, outputs, predicted_depth
            torch.cuda.empty_cache() if self.device.type == 'cuda' else gc.collect()
            
            # Normalize depth values
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            return depth_map
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM error. Trying with smaller image size...")
                self.max_image_size = self.max_image_size // 2
                image = self.resize_image(image)
                return self.estimate_depth(image)
            raise e

    def generate_point_cloud(self, depth_map, image):
        """Generate point cloud with real-world scaling"""
        height, width = depth_map.shape
        
        # Calculate scaling factor based on camera intrinsics
        focal_length_px = self.focal_length
        sensor_width_px = width  # Assuming the image width corresponds to the sensor width
        scale_factor = (self.sensor_width_mm / 1000) / sensor_width_px  # Convert to meters
        
        # Create coordinate grid
        x, y = np.meshgrid(
            np.linspace(-width / 2, width / 2, width),
            np.linspace(-height / 2, height / 2, height)
        )
        
        # Convert pixel coordinates to real-world coordinates
        x = x * scale_factor
        y = y * scale_factor
        
        # Scale depth to real-world units (meters)
        if self.known_distance is not None:
            # Calibrate depth using a known distance
            depth_map *= (self.known_distance / depth_map.max())
        else:
            # Default scaling (assume depth is in meters)
            depth_map *= 10.0  # Arbitrary scaling factor
        
        # Stack coordinates
        points = np.stack([x, y, depth_map], axis=-1)
        points = points.reshape(-1, 3)
        
        # Remove invalid points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        
        return points

    def generate_mesh(self, points):
        """Generate mesh using Poisson Surface Reconstruction"""
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Estimate normals (required for Poisson reconstruction)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Perform Poisson Surface Reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Remove low-density vertices (optional)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh

    def process_image(self, image_path):
        """Process image to 3D with advanced features"""
        try:
            # Load and process image
            image = self.load_image(image_path)
            
            # Get depth map
            depth_map = self.estimate_depth(image)
            
            # Generate point cloud with real-world scaling
            points = self.generate_point_cloud(depth_map, image)
            
            # Generate mesh using advanced techniques
            mesh = self.generate_mesh(points)
            
            # Reorient axes: Z should be up
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[:, [0, 2, 1]])
            
            # Calculate dimensions of the 3D model
            vertices = np.asarray(mesh.vertices)
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            dimensions = max_coords - min_coords
            print(f"3D Model Dimensions (Width, Depth, Height): {dimensions} meters")
            
            # Clear memory
            gc.collect()
            
            return mesh
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

def main():
    # Initialize converter with advanced settings
    converter = AdvancedImageTo3D(
        max_image_size=384,
        focal_length=1000,  # Focal length in pixels
        sensor_width_mm=36.0,  # Sensor width in millimeters
        known_distance=100.0  # Known distance in meters (e.g., a person standing 5 meters away)
    )
    
    # Process image
    image_path = "/home/anushtup/git-repos/rl-bullet-gym/image3d/images/liberty.jpg"  # Update with your image path
    print("Processing image...")
    mesh = converter.process_image(image_path)
    
    # Convert Open3D mesh to PyVista mesh
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # PyVista requires faces to be in a specific format (e.g., [n, v1, v2, v3])
    faces_pv = np.hstack([np.ones((faces.shape[0], 1), dtype=int) * 3, faces])
    
    # Create PyVista mesh
    mesh_pv = pv.PolyData(vertices, faces_pv)
    
    # Visualize the mesh using PyVista
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, color='lightgray', show_edges=True, opacity=0.7, smooth_shading=True)
    plotter.show_grid()
    plotter.show_axes()
    plotter.show()

if __name__ == "__main__":
    main()