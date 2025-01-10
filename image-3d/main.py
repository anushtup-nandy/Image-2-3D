import argparse
from depth_estimation import DepthEstimator
from point_cloud import PointCloudGenerator
from mesh_generation import MeshGenerator
from texture_mapping import TextureMapper
from mesh_analysis import MeshAnalyzer
from export_model import ModelExporter
from visualization import Visualizer
from PIL import Image
import warnings
import numpy as np

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore", message="Some weights of DPTForDepthEstimation were not initialized")

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a 3D model from an image.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--human_distance", type=float, required=True, help="Distance of the human from the object (in meters).")
    parser.add_argument("--focal_length", type=float, default=1000, help="Focal length of the camera (in pixels).")
    parser.add_argument("--sensor_width", type=float, default=36.0, help="Sensor width of the camera (in millimeters).")
    parser.add_argument("--export_path", type=str, default="output.obj", help="Path to export the 3D model.")
    parser.add_argument("--mesh_type", type=str, default="poisson", choices=["poisson", "ball_pivot"],
                        help="Mesh generation method: 'poisson' or 'ball_pivot'.")
    args = parser.parse_args()

    # Load image
    image = Image.open(args.image_path)

    # Estimate depth
    depth_estimator = DepthEstimator()
    depth_map = depth_estimator.estimate_depth(image)

    # Generate point cloud
    point_cloud_generator = PointCloudGenerator(args.focal_length, args.sensor_width, args.human_distance)
    points = point_cloud_generator.generate_point_cloud(depth_map, image)

    # Generate mesh
    mesh_generator = MeshGenerator(mesh_type=args.mesh_type)
    mesh = mesh_generator.generate_mesh(points)

    # Map texture
    texture_mapper = TextureMapper()
    mesh = texture_mapper.map_texture(mesh, image)

    # Analyze mesh
    mesh_analyzer = MeshAnalyzer()
    surface_area = mesh_analyzer.calculate_surface_area(mesh)
    volume = mesh_analyzer.calculate_volume(mesh)
    width, depth, height = mesh_analyzer.calculate_dimensions(mesh)

    # Print results
    print(f"Surface Area: {surface_area} m²")
    if volume is not None:
        print(f"Volume: {volume} m³")
    print(f"Dimensions (Width x Depth x Height): {width:.2f} m x {depth:.2f} m x {height:.2f} m")

    # Export model
    model_exporter = ModelExporter()
    model_exporter.export_model(mesh, args.export_path, format="obj")

    # Visualize mesh
    visualizer = Visualizer()
    visualizer.visualize_mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

if __name__ == "__main__":
    main()