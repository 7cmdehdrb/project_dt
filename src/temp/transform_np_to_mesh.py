from geometry_msgs.msg import Point
from shape_msgs.msg import Mesh, MeshTriangle
import numpy as np
import carb


def as_shape_msgs_mesh_from_csv(verts, faces):
    """
    Convert a trimesh.Trimesh to shape_msgs/Mesh with optional per-axis scale.
    Ensures triangles only.
    """

    vertices = []
    triangles = []

    for vx, vy, vz in verts:
        vtxt = f'{{"x":{float(vx)},"y":{float(vy)},"z":{float(vz)}}}'
        vertices.append(vtxt)

    for i0, i1, i2 in faces:
        ttxt = f'{{"vertex_indices":[{i0},{i1},{i2}]}}'
        triangles.append(ttxt)

    return vertices, triangles


def setup(db: og.Database):
    pass


def cleanup(db: og.Database):
    pass


def compute(db: og.Database):
    verts = db.inputs.verts  # (N,3) float32
    faces = db.inputs.faces  # (M,3) int32

    if verts is None or faces is None:
        carb.log_warn("verts or faces input is None.")
        return False

    vertices, triangles = as_shape_msgs_mesh_from_csv(verts, faces)

    db.outputs.vertices = vertices
    db.outputs.triangles = triangles

    carb.log_info(f"Output {len(vertices)} vertices and {len(triangles)} triangles.")

    return True
