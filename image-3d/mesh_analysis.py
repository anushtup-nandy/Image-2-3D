import numpy as np
import open3d as o3d

class MeshAnalyzer:
    def calculate_surface_area(self, mesh):
        """Calculate the surface area of the mesh."""
        return np.asarray(mesh.get_surface_area())

    def calculate_volume(self, mesh):
        """Calculate the volume of the mesh if it is watertight."""
        if mesh.is_watertight():
            return np.asarray(mesh.get_volume())
        else:
            print("Warning: Mesh is not watertight. Volume cannot be computed.")
            return None

    def compute_normals(self, mesh):
        """Compute vertex normals for the mesh."""
        mesh.compute_vertex_normals()
        return mesh

    def detect_sharp_edges(self, mesh, angle_threshold=30):
        """Detect sharp edges in the mesh."""
        mesh.compute_vertex_normals()
        edges = np.asarray(mesh.get_non_manifold_edges())
        return edges

    def calculate_dimensions(self, mesh):
        """Calculate the height, depth, and width of the mesh."""
        vertices = np.asarray(mesh.vertices)
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # Calculate dimensions
        width = max_coords[0] - min_coords[0]  # Along x-axis
        depth = max_coords[1] - min_coords[1]  # Along y-axis
        height = max_coords[2] - min_coords[2]  # Along z-axis
        
        return width, depth, height