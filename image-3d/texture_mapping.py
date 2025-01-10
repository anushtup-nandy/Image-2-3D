import numpy as np
import open3d as o3d

class TextureMapper:
    def map_texture(self, mesh, image):
        """Map the image texture onto the mesh."""
        # Convert image to Open3D texture
        image = np.asarray(image)
        texture = o3d.geometry.Image(image)

        # Assign texture coordinates (UV mapping)
        vertices = np.asarray(mesh.vertices)
        uv_coords = (vertices[:, :2] - vertices[:, :2].min(axis=0)) / (vertices[:, :2].max(axis=0) - vertices[:, :2].min(axis=0))
        mesh.triangle_uvs = o3d.utility.Vector2dVector(np.vstack([uv_coords, uv_coords, uv_coords]))

        # Assign texture
        mesh.textures = [texture]
        return mesh