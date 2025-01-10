import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

class ImageTo3D:
    def __init__(self):
        """Initialize the 3D conversion pipeline with necessary models"""
        # Initialize the depth estimation model
        self.depth_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.depth_model.to(self.device)

    def load_image(self, image_path):
        """Load and preprocess the image"""
        # Load image using PIL
        image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image

    def estimate_depth(self, image):
        """Estimate depth map from the input image"""
        # Prepare image for the model
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Normalize depth values
        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        return depth_map[0, 0].cpu().numpy()

    def generate_point_cloud(self, image, depth_map):
        """Generate 3D point cloud from image and depth map"""
        # Get image dimensions
        height, width = depth_map.shape
        
        # Create coordinate grid
        x_coords = np.linspace(0, width-1, width)
        y_coords = np.linspace(0, height-1, height)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Create point cloud
        points = np.zeros((height, width, 3))
        points[:,:,0] = x_grid
        points[:,:,1] = y_grid
        points[:,:,2] = depth_map
        
        # Reshape to Nx3 array
        points = points.reshape(-1, 3)
        
        return points

    def get_3d_dimensions(self, point_cloud):
        """Calculate dimensions of the 3D point cloud"""
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
        dimensions = max_coords - min_coords
        
        return {
            'width': dimensions[0],
            'height': dimensions[1],
            'depth': dimensions[2]
        }

    def process_image(self, image_path):
        """Main processing pipeline"""
        # Load image
        image = self.load_image(image_path)
        
        # Estimate depth
        depth_map = self.estimate_depth(image)
        
        # Generate point cloud
        point_cloud = self.generate_point_cloud(image, depth_map)
        
        # Get dimensions
        dimensions = self.get_3d_dimensions(point_cloud)
        
        return {
            'point_cloud': point_cloud,
            'dimensions': dimensions,
            'depth_map': depth_map
        }

# Example usage
def main():
    # Initialize converter
    converter = ImageTo3D()
    
    # Process image
    image_path = "/home/anushtup/git-repos/rl-bullet-gym/image3d/images/building-nyc.jpg"
    result = converter.process_image(image_path)
    
    # Print dimensions
    print("3D Dimensions:")
    print(f"Width: {result['dimensions']['width']:.2f}")
    print(f"Height: {result['dimensions']['height']:.2f}")
    print(f"Depth: {result['dimensions']['depth']:.2f}")
    
    # Optionally save the point cloud
    np.save('point_cloud.npy', result['point_cloud'])
    
if __name__ == "__main__":
    main()