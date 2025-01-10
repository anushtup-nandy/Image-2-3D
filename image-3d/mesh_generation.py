import open3d as o3d
import numpy as np

class MeshGenerator:
    def __init__(self, mesh_type="poisson"):
        """Initialize the mesh generator with the specified type."""
        self.mesh_type = mesh_type

    def generate_mesh(self, points):
        """Generate a mesh from the point cloud using the specified method."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()

        if self.mesh_type == "poisson":
            mesh = self._generate_mesh_poisson(pcd)
        elif self.mesh_type == "ball_pivot":
            mesh = self._generate_mesh_ball_pivot(pcd)
        else:
            raise ValueError(f"Unsupported mesh type: {self.mesh_type}")

        # Clean the mesh
        mesh = mesh.remove_non_manifold_edges()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_duplicated_triangles()
        mesh = mesh.remove_duplicated_vertices()

        return mesh

    def _generate_mesh_poisson(self, pcd):
        """Generate a mesh using Poisson Surface Reconstruction."""
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        return mesh

    def _generate_mesh_ball_pivot(self, pcd):
        """Generate a mesh using Ball Pivoting Algorithm (BPA)."""
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist  # Adjust the radius for sharper or smoother results

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2])
        )

        return mesh