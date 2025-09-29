from pxr import Usd, UsdGeom
import numpy as np
import omni.usd
import carb


def triangulate(counts, indices):
    """
    Convert faceVertexCounts and faceVertexIndices to triangle indices.
    :return: (M,3) int32 numpy array
    """
    tris = []
    i = 0
    for c in counts:
        if c == 3:
            tris.append(indices[i : i + 3])
        elif c > 3:
            base = indices[i]
            for k in range(1, c - 1):
                tris.append([base, indices[i + k], indices[i + k + 1]])
        i += c
    return np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)


def extract_mesh(prim_path):
    """
    Extract verts and faces as numpy arrays from the given Mesh prim path.
    :param prim_path: string (e.g., "/meshes/shoulder_0/mesh")
    :return: verts (N,3 float32), faces (M,3 int32)
    """
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or prim.GetTypeName() != "Mesh":
        raise RuntimeError(f"{prim_path} is not a Mesh prim")

    mesh = UsdGeom.Mesh(prim)

    verts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)  # (N,3)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)

    faces = triangulate(counts, indices)  # (M,3)

    return verts, faces


def downsample_vertex_clustering(
    verts: np.ndarray, faces: np.ndarray, voxel_size=0.01, reducer="mean"
):
    """
    verts: (N,3) float
    faces: (M,3) int
    voxel_size: float, 클러스터링 격자 크기
    reducer: 'mean'|'first'|'median'
    return: (new_verts, new_faces)
    """
    if len(verts) == 0 or len(faces) == 0:
        return verts.copy(), faces.copy()

    keys = np.floor(verts / voxel_size).astype(np.int64)
    # 고유 클러스터와 역매핑
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)

    # 클러스터 대표점 계산
    new_verts = np.zeros((uniq.shape[0], 3), dtype=verts.dtype)
    if reducer == "mean":
        # 각 클러스터별 평균
        for i in range(uniq.shape[0]):
            new_verts[i] = verts[inv == i].mean(axis=0)
    elif reducer == "median":
        for i in range(uniq.shape[0]):
            new_verts[i] = np.median(verts[inv == i], axis=0)
    else:  # 'first'
        # 각 클러스터의 첫 정점 선택
        first_idx = np.zeros(uniq.shape[0], dtype=np.int64)
        # inv의 첫 등장 인덱스 구하기
        seen = -np.ones(uniq.shape[0], dtype=np.int64)
        for idx, cid in enumerate(inv):
            if seen[cid] < 0:
                seen[cid] = idx
        first_idx = seen
        new_verts = verts[first_idx]

    # faces 재인덱싱
    new_faces = inv[faces]

    # 퇴화 면(동일 정점 포함) 제거
    mask = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    new_faces = new_faces[mask]

    # 사용되지 않는 정점 정리(선택적)
    used = np.unique(new_faces.reshape(-1))
    remap = -np.ones(new_verts.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size)
    new_verts = new_verts[used]
    new_faces = remap[new_faces]

    return new_verts, new_faces


verts = np.empty((0, 3), dtype=np.float32)
faces = np.empty((0, 3), dtype=np.int32)


def setup(db: og.Database):
    global verts, faces

    prim_path = db.inputs.prim_path
    temp_verts, temp_faces = extract_mesh(prim_path)

    verts, faces = downsample_vertex_clustering(
        temp_verts, temp_faces, voxel_size=0.1, reducer="mean"
    )
    carb.log_info(
        f"Extracted {len(verts)} vertices and {len(faces)} faces from {prim_path}."
    )


def cleanup(db: og.Database):
    global verts, faces
    verts = np.empty((0, 3), dtype=np.float32)
    faces = np.empty((0, 3), dtype=np.int32)


def compute(db: og.Database):
    global verts, faces

    if verts.size == 0 or faces.size == 0:
        carb.log_warn("No mesh data extracted.")
        return False

    db.outputs.verts = verts
    db.outputs.faces = faces

    carb.log_info(f"Output {len(verts)} vertices and {len(faces)} faces.")

    return True
