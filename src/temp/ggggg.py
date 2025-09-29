from pxr import Usd, UsdGeom
import numpy as np
import omni.usd

import csv

def write_mesh_to_csv(verts, faces, vfilename='verts.csv', ffilename='faces.csv'):
    """
    Writes verts and faces arrays to separate CSV files.

    Parameters:
        verts (ndarray): (n, 3) array of vertex coordinates.
        faces (ndarray): (n, 3) array of face indices.
        vfilename (str): Output CSV file name for vertices.
        ffilename (str): Output CSV file name for faces.
    """
    with open(vfilename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header for vertices
        writer.writerow(['verts_x', 'verts_y', 'verts_z'])
        for vert in verts:
            writer.writerow(vert)

    with open(ffilename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header for faces
        writer.writerow(['faces_v1', 'faces_v2', 'faces_v3'])
        for face in faces:
            writer.writerow(face)
            
    print(f"Wrote {len(verts)} vertices to {vfilename} and {len(faces)} faces to {ffilename}.")
             
def triangulate(counts, indices):
    """
    faceVertexCounts, faceVertexIndices를 삼각형 인덱스로 변환
    :return: (M,3) int32 numpy 배열
    """
    tris = []
    i = 0
    for c in counts:
        if c == 3:
            tris.append(indices[i:i+3])
        elif c > 3:
            base = indices[i]
            for k in range(1, c-1):
                tris.append([base, indices[i+k], indices[i+k+1]])
        i += c
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0,3), dtype=np.int32)

def extract_mesh(prim_path):
    """
    주어진 Mesh prim 경로에서 verts, faces를 numpy 배열로 추출
    :param prim_path: string (예: "/meshes/shoulder_0/mesh")
    :return: verts (N,3 float32), faces (M,3 int32)
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"{prim_path} is not a Mesh prim")

    mesh = UsdGeom.Mesh(prim)

    verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)   # (N,3)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    faces = triangulate(counts, indices)  # (M,3)

    return verts, faces

# === 사용 예시 ===
verts, faces = extract_mesh("/ur5e/shoulder_link/visuals/shoulder/mesh")
print("verts:", verts.shape)
print("faces:", faces.shape)

    
    
#$write_mesh_to_csv(verts, faces, vfilename='/home/min/7cmdehdrb/project_th/verts.csv', ffilename='/home/min/7cmdehdrb/project_th/faces.csv')
