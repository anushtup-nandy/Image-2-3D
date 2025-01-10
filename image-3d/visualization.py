import pyvista as pv
import numpy as np

class Visualizer:
    def visualize_mesh(self, vertices, faces, texture=None):
        """Visualize the mesh using PyVista."""
        faces_pv = np.hstack([np.ones((faces.shape[0], 1), dtype=int) * 3, faces])
        mesh_pv = pv.PolyData(vertices, faces_pv)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh_pv, color='lightgray', show_edges=True, opacity=0.7, smooth_shading=True)
        plotter.show_grid()
        plotter.show_axes()
        plotter.show()