import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def visualize_point_cloud(npy_file_path, downsample_ratio=0.5):
    """
    Visualize a point cloud from a .npy file
    Args:
        npy_file_path: Path to the .npy file containing point cloud data
        downsample_ratio: Ratio to downsample points (0.5 means use 50% of points)
    """
    # Load the point cloud data
    points = np.load(npy_file_path)
    
    # Downsample points for better performance
    num_points = len(points)
    indices = np.random.choice(num_points, int(num_points * downsample_ratio), replace=False)
    points = points[indices]
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Estimate normals for better visualization
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Create coordinate frame for reference
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    
    # Visualization settings
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Viewer")
    
    # Add geometry
    vis.add_geometry(pcd)
    vis.add_geometry(coordinate_frame)
    
    # Set up render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark gray background
    opt.point_size = 2.0
    
    # Set up initial camera view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    
    print("Controls:")
    print("- Left click + drag: Rotate")
    print("- Right click + drag: Pan")
    print("- Mouse wheel: Zoom")
    print("- Press 'H' to see more keyboard controls")
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_with_color(npy_file_path, downsample_ratio=0.5):
    """
    Visualize point cloud with colors based on height (z-coordinate)
    Args:
        npy_file_path: Path to the .npy file containing point cloud data
        downsample_ratio: Ratio to downsample points
    """
    # Load and downsample points
    points = np.load(npy_file_path)
    num_points = len(points)
    indices = np.random.choice(num_points, int(num_points * downsample_ratio), replace=False)
    points = points[indices]
    
    # Create color map based on height (z-coordinate)
    z_vals = points[:, 2]
    colors = plt.cm.viridis((z_vals - z_vals.min()) / (z_vals.max() - z_vals.min()))[:, :3]
    
    # Create and setup point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def main():
    """Main function demonstrating both visualization methods"""
    npy_file = "point_cloud.npy"  # Update this path to your .npy file
    
    print("1. Basic Visualization")
    visualize_point_cloud(npy_file)
    
    print("\n2. Height-colored Visualization")
    visualize_with_color(npy_file)

if __name__ == "__main__":
    main()