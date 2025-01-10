import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_point_cloud_matplotlib(npy_file_path, downsample_ratio=0.5):
    """
    Visualize point cloud using Matplotlib
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
    
    # Extract x, y, z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points with color based on depth
    scatter = ax.scatter(x, y, z, 
                        c=z,  # Color based on z-coordinate
                        cmap='viridis',
                        s=1,  # Point size
                        alpha=0.5)  # Transparency
    
    # Add a color bar
    plt.colorbar(scatter, label='Depth')
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Set title
    plt.title('Point Cloud Visualization')
    
    # Add interaction instructions
    print("Controls:")
    print("- Click and drag to rotate")
    print("- Right click and drag to zoom")
    print("- Middle click and drag to pan")
    
    # Show the plot
    plt.show()

def visualize_point_cloud_subplots(npy_file_path, downsample_ratio=0.5):
    """
    Visualize point cloud from different angles using subplots
    Args:
        npy_file_path: Path to the .npy file containing point cloud data
        downsample_ratio: Ratio to downsample points
    """
    # Load and downsample points
    points = np.load(npy_file_path)
    num_points = len(points)
    indices = np.random.choice(num_points, int(num_points * downsample_ratio), replace=False)
    points = points[indices]
    
    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Top view
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
    ax1.set_title('Top View (X-Y)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(scatter1, ax=ax1)
    
    # Front view
    ax2 = fig.add_subplot(132)
    scatter2 = ax2.scatter(x, z, c=z, cmap='viridis', s=1, alpha=0.5)
    ax2.set_title('Front View (X-Z)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    plt.colorbar(scatter2, ax=ax2)
    
    # Side view
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(y, z, c=z, cmap='viridis', s=1, alpha=0.5)
    ax3.set_title('Side View (Y-Z)')
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    plt.colorbar(scatter3, ax=ax3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function demonstrating both visualization methods"""
    npy_file = "/home/anushtup/git-repos/rl-bullet-gym/image3d/point_cloud.npy"  # Update this path to your .npy file
    
    print("1. 3D Interactive Visualization")
    visualize_point_cloud_matplotlib(npy_file)
    
    print("\n2. Multi-view Visualization")
    visualize_point_cloud_subplots(npy_file)

if __name__ == "__main__":
    main()