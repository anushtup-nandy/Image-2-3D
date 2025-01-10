import open3d as o3d

class ModelExporter:
    def export_model(self, mesh, file_path, format="obj"):
        """Export the 3D model to a file."""
        if format == "obj":
            o3d.io.write_triangle_mesh(file_path, mesh, write_vertex_normals=True)
        elif format == "stl":
            o3d.io.write_triangle_mesh(file_path, mesh)
        elif format == "ply":
            o3d.io.write_triangle_mesh(file_path, mesh)
        else:
            raise ValueError(f"Unsupported format: {format}")