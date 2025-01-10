import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import pyvista as pv
from scipy.spatial import Delaunay
import gc  # Garbage collector

class MemoryEfficientImageTo3D:
    def __init__(self, max_image_size=384, depth_scale=50.0):
        """Initialize with memory constraints
        Args:
            max_image_size: Maximum size for image's longest side
            depth_scale: Scaling factor for depth values (in meters)
        """
        self.max_image_size = max_image_size
        self.depth_scale = depth_scale  # Scaling factor for depth (in meters)
        
        # Initialize depth model with memory efficiency options
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            "Intel/dpt-large",  # Using smaller hybrid model instead of large
            torch_dtype=torch.float16  # Use half precision
        )
        
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-large",  # Using smaller hybrid model
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

    def generate_point_cloud(self, depth_map, downsample_factor=2):
        """Generate point cloud with downsampling option"""
        height, width = depth_map.shape
        
        # Downsample depth map
        if downsample_factor > 1:
            height = height // downsample_factor
            width = width // downsample_factor
            depth_map = cv2.resize(depth_map, (width, height))
        
        # Create coordinate grid
        x, y = np.meshgrid(
            np.linspace(-1, 1, width),
            np.linspace(-1, 1, height)
        )
        
        # Stack coordinates
        points = np.stack([x, y, depth_map], axis=-1)
        points = points.reshape(-1, 3)
        
        # Remove invalid points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        
        return points

    def process_image(self, image_path, downsample_factor=2):
        """Process image to 3D with memory efficiency"""
        try:
            # Load and process image
            image = self.load_image(image_path)
            
            # Get depth map
            depth_map = self.estimate_depth(image)
            
            # Generate point cloud
            points = self.generate_point_cloud(depth_map, downsample_factor)
            
            # Scale depth to real-world units (meters)
            points[:, 2] *= self.depth_scale  # Scale depth (z-axis)
            
            # Reorient axes: Z should be up
            # Swap y and z axes
            points[:, [1, 2]] = points[:, [2, 1]]
            
            # Create mesh (with reduced point count)
            if len(points) > 10000:  # If too many points, subsample
                indices = np.random.choice(len(points), 10000, replace=False)
                points_for_mesh = points[indices]
            else:
                points_for_mesh = points
                
            # Create triangulation
            tri = Delaunay(points_for_mesh[:, :2])
            triangles = tri.simplices
            
            # Convert triangles to PyVista format
            # Each triangle should be prefixed with the number of vertices (3)
            triangles_pv = np.hstack(
                [np.ones((triangles.shape[0], 1), dtype=int) * 3, triangles]
            )
            
            # Clear memory
            gc.collect()
            
            return points_for_mesh, triangles_pv
            
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise

def main():
    # Initialize converter with memory constraints
    converter = MemoryEfficientImageTo3D(max_image_size=384, depth_scale=100.0)  # Depth scale in meters
    
    # Process image
    image_path = "/home/anushtup/git-repos/rl-bullet-gym/image3d/images/liberty.jpg"  # Update with your image path
    print("Processing image...")
    points, triangles = converter.process_image(image_path, downsample_factor=2)
    
    # Calculate dimensions of the 3D model
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    dimensions = max_coords - min_coords
    print(f"3D Model Dimensions (Width, Depth, Height): {dimensions} meters")
    
    # Initialize simple viewer
    plotter = pv.Plotter()
    
    # Create mesh
    mesh = pv.PolyData(points, triangles)
    
    # Add to plotter
    plotter.add_mesh(mesh, 
                    color='lightgray',
                    show_edges=True,
                    opacity=0.7,
                    smooth_shading=True)
    
    # Show grid and axes
    plotter.show_grid()
    plotter.show_axes()
    
    # Display
    plotter.show()

if __name__ == "__main__":
    main()