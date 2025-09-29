from pxr import Usd, UsdGeom, Gf
import omni.usd
from omni.isaac.core.prims import XFormPrim
import numpy as np
import carb


DATA = {}


# --- 4x4 월드 행렬을 넘파이로 반환 ---
def get_world_matrix_np(prim_path: str, time=Usd.TimeCode.Default()) -> np.ndarray:
    """Return the world transform of a prim as a 4x4 numpy array (float64)."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    m = xform.ComputeLocalToWorldTransform(time)  # Gf.Matrix4d
    return np.array(m, dtype=np.float64)  # (4,4)


# --- 4x4 행렬 → (pos, quat) 추출 (스케일/전단 제거: Polar Decomposition) ---
def mat4_to_pos_quat(m44: np.ndarray):
    """
    Extract (position, quaternion(x,y,z,w)) from a 4x4 transform by
    removing scale/shear via polar decomposition (R = U V^T).
    """
    if m44.shape != (4, 4):
        raise ValueError("m44 must be 4x4")
    T = m44[:3, 3].astype(np.float64)
    A = m44[:3, :3].astype(np.float64)
    # Polar decomposition: A = R * S  (R: rotation, S: symmetric)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    # Ensure right-handed (det=+1)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    # Rotation matrix -> quaternion (xyzw)
    qw = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-20)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-20)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-20)
    return (T[0], T[1], T[2]), (qx, qy, qz, qw)


class MeshFinder(object):
    def __init__(self, prim_path: str, downsample_voxel_size=0.1, scale=1.0):
        """
        prim_path: USD Mesh prim path (e.g., "/meshes/shoulder_0/mesh")
        downsample_voxel_size: float, voxel size used for vertex clustering
        scale: float or (3,) array-like.
               - float -> uniform scaling
               - (sx, sy, sz) -> per-axis scaling
        """
        self._prim_path = prim_path
        self._downsample_voxel_size = float(downsample_voxel_size)
        self._scale = self._normalize_scale(scale)

        self._verts, self._faces = self._extract_mesh(prim_path)

    @property
    def verts(self):
        return self._verts

    @property
    def faces(self):
        return self._faces

    def _normalize_scale(self, scale):
        s = np.asarray(scale, dtype=np.float32)
        if s.ndim == 0:
            s = np.array([float(s)] * 3, dtype=np.float32)
        elif s.shape == (3,):
            s = s.astype(np.float32)
        else:
            raise ValueError("scale must be a float or a length-3 array-like")
        return s

    def _apply_scale(self, verts):
        # verts: (N,3), self._scale: (3,)
        return verts * self._scale.reshape(1, 3)

    def _extract_mesh(self, prim_path):
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

        faces = self._triangulate(counts, indices)  # (M,3)

        verts = self._apply_scale(verts)

        downsampled_verts, downsampled_faces = self._downsample_vertex_clustering(
            verts, faces, voxel_size=self._downsample_voxel_size, reducer="mean"
        )

        return downsampled_verts, downsampled_faces

    def _triangulate(self, counts, indices):
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
        return (
            np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
        )

    def _downsample_vertex_clustering(
        self, verts, faces, voxel_size: float, reducer="mean"
    ):
        """
        verts: (N,3) float
        faces: (M,3) int
        voxel_size: float, clustering grid size (same units as verts after scaling)
        reducer: 'mean'|'first'|'median'
        return: (new_verts, new_faces)
        """
        if len(verts) == 0 or len(faces) == 0:
            return verts.copy(), faces.copy()

        keys = np.floor(verts / voxel_size).astype(np.int64)
        # unique clusters and inverse map
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)

        # representative for each cluster
        new_verts = np.zeros((uniq.shape[0], 3), dtype=verts.dtype)
        if reducer == "mean":
            for i in range(uniq.shape[0]):
                new_verts[i] = verts[inv == i].mean(axis=0)
        elif reducer == "median":
            for i in range(uniq.shape[0]):
                new_verts[i] = np.median(verts[inv == i], axis=0)
        else:  # 'first'
            seen = -np.ones(uniq.shape[0], dtype=np.int64)
            for idx, cid in enumerate(inv):
                if seen[cid] < 0:
                    seen[cid] = idx
            new_verts = verts[seen]

        # reindex faces
        new_faces = inv[faces]

        # drop degenerate faces
        mask = (
            (new_faces[:, 0] != new_faces[:, 1])
            & (new_faces[:, 1] != new_faces[:, 2])
            & (new_faces[:, 2] != new_faces[:, 0])
        )
        new_faces = new_faces[mask]

        # compact unused vertices
        used = np.unique(new_faces.reshape(-1))
        remap = -np.ones(new_verts.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size)
        new_verts = new_verts[used]
        new_faces = remap[new_faces]

        return new_verts, new_faces

    @staticmethod
    def find_mesh_paths(root_path: str):
        """
        Returns a list of all Mesh prim paths under the given prim_path.
        Covers reference, instanceable, and instance-proxy cases.
        """
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise RuntimeError(f"Invalid prim path: {root_path}")

        # Attempt to load in case there is a payload (ignore errors)
        try:
            root.Load()
            stage.Load(root_path)
        except Exception:
            pass

        paths = set()

        def scan(prim_range):
            for p in prim_range:
                if p.GetTypeName() == "Mesh":
                    paths.add(p.GetPath().pathString)

        scan(Usd.PrimRange(root, Usd.PrimDefaultPredicate))

        if hasattr(Usd, "TraverseInstanceProxies"):
            scan(
                Usd.PrimRange(
                    root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
                )
            )

        if root.IsInstance():
            proto = root.GetPrototype()
            if proto:
                scan(Usd.PrimRange(proto, Usd.PrimDefaultPredicate))

        if root.IsInstanceProxy():
            pinp = root.GetPrimInPrototype()
            if pinp:
                scan(Usd.PrimRange(pinp, Usd.PrimDefaultPredicate))

        return sorted(paths)


class MeshCombiner(object):
    def __init__(self, meshes, poses):
        self._meshes = [(m.verts, m.faces) for m in meshes]  # list of (verts, faces)
        self._poses = poses

        self._verts, self._faces = self._merge_meshes_with_poses(
            self._meshes, self._poses
        )

    @property
    def verts(self):
        return self._verts

    @property
    def faces(self):
        return self._faces

    def _apply_pose(self, verts, pose4x4):
        # verts: (N,3), pose: (4,4)
        N = verts.shape[0]
        homo = np.c_[verts, np.ones((N, 1), dtype=verts.dtype)]
        out = (homo @ pose4x4.T)[:, :3]
        return out

    def _merge_meshes_with_poses(self, meshes, poses):
        """
        meshes: list of (verts(Ni,3), faces(Mi,3))
        poses:  list of (4,4) world transforms for each mesh
        return: (merged_verts, merged_faces)
        """
        assert len(meshes) == len(poses)
        all_verts = []
        all_faces = []
        offset = 0
        for (v, f), T in zip(meshes, poses):
            v_w = self._apply_pose(v, T)
            all_verts.append(v_w)
            all_faces.append(f + offset)
            offset += v.shape[0]

        merged_verts = np.vstack(all_verts).astype(np.float64, copy=False)
        merged_faces = np.vstack(all_faces).astype(np.int32, copy=False)
        return merged_verts, merged_faces

    @staticmethod
    def mesh_serialization(verts, faces):
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


class WorldTransformer(object):
    @staticmethod
    def transform_from_pos_quat(pos, quat):
        """
        Convert (position, quaternion(x,y,z,w)) -> 4x4 homogeneous transform matrix.
        (병합에는 사용하지 않지만, 필요 시 남겨둡니다.)
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)
        n = np.sqrt(x * x + y * y + z * z + w * w)
        if n == 0:
            raise ValueError("Quaternion has zero norm.")
        x, y, z, w = x / n, y / n, z / n, w / n

        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    @staticmethod
    def transform_from_pos_quat(pos, quat):
        """
        Convert (position, quaternion(x,y,z,w)) -> 4x4 homogeneous transform matrix.

        Args:
            pq: tuple (pos, quat)
                - pos: iterable of 3 numbers (x, y, z)
                - quat: iterable of 4 numbers (x, y, z, w)
            dtype: numpy dtype of the returned matrix

        Returns:
            4x4 homogeneous transform matrix (numpy.ndarray)
        """
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        x, y, z, w = np.asarray(quat, dtype=np.float64).reshape(4)

        # Normalize quaternion
        n = np.sqrt(x * x + y * y + z * z + w * w)
        if n == 0:
            raise ValueError("Quaternion has zero norm.")
        x, y, z, w = x / n, y / n, z / n, w / n

        # Rotation matrix
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

        # Homogeneous transform
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos
        return T

    @staticmethod
    def transform_euler_to_quat(x, y, z):
        """
        Convert Euler angles (in radians) to a quaternion.
        The input is expected to be in the order of roll (X), pitch (Y), yaw (Z).
        """
        roll, pitch, yaw = x, y, z

        roll = np.deg2rad(roll)
        pitch = np.deg2rad(pitch)
        yaw = np.deg2rad(yaw)

        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw

    @staticmethod
    def get_world_pose_from_xform(prim_path):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # 행렬 → 위치, 쿼터니언
        translation = m.ExtractTranslation()
        rotation = Gf.Quatd(m.ExtractRotationQuat())
        pos = (translation[0], translation[1], translation[2])
        quat = (
            rotation.GetImaginary()[0],
            rotation.GetImaginary()[1],
            rotation.GetImaginary()[2],
            rotation.GetReal(),
        )

        return pos, quat


# === get_mesh: 4x4 행렬 기반 병합으로 수정 ===
def get_mesh(db: og.Database):
    global DATA

    # STEP 0. Get inputs
    prim_path = db.inputs.prim_path  # str
    voxel_size = db.inputs.downsample_voxel_size  # float
    scale = db.inputs.scale  # float or (3,) array-like

    print(f"Start to get mesh under {prim_path}.")

    # STEP 1. Find all mesh paths under the given prim_path
    mesh_paths = MeshFinder.find_mesh_paths(prim_path)

    # STEP 2. Get world matrix of the target prim (as reference frame)
    T_global = get_world_matrix_np(prim_path)  # 4x4 (float64)
    T_global_inv = np.linalg.inv(T_global)

    # (옵션) DATA 저장용 pos/quat (Polar로 스케일/전단 제거)
    pos_global, quat_global = mat4_to_pos_quat(T_global)

    # STEP 3-0. Initialize lists
    mesh_list = []
    pose_list = []

    # STEP 3-1. For each mesh path, extract and downsample the mesh, get relative world pose
    for path in mesh_paths:
        # STEP 3-2. Append MeshFinder (로컬 메시)
        mesh = MeshFinder(path, downsample_voxel_size=voxel_size, scale=scale)

        # STEP 3-3. Compute transform from global to this mesh (4x4 full matrix)
        T_mesh = get_world_matrix_np(path)  # 4x4 (float64)
        T_rel = T_global_inv @ T_mesh

        mesh_list.append(mesh)
        pose_list.append(T_rel)

    # STEP 4. Combine all meshes into one mesh in the global frame
    combiner = MeshCombiner(mesh_list, pose_list)
    verts, faces = combiner.verts, combiner.faces
    verts, faces = MeshCombiner.mesh_serialization(combiner.verts, combiner.faces)

    # STEP 5. Store the combined mesh (pos/quat는 Polar로 정제된 값)
    DATA[prim_path] = {
        "verts": verts,
        "faces": faces,
        "position": {
            "x": float(pos_global[0]),
            "y": float(pos_global[1]),
            "z": float(pos_global[2]),
        },
        "orientation": {
            "x": float(quat_global[0]),
            "y": float(quat_global[1]),
            "z": float(quat_global[2]),
            "w": float(quat_global[3]),
        },
    }

    print(
        f"Combined mesh {prim_path} has {len(verts)} vertices and {len(faces)} faces."
    )


def update_pose(db: og.Database):
    global DATA

    # STEP 0. Get inputs
    prim_path = db.inputs.prim_path  # str

    pos, quat = WorldTransformer.get_world_pose_from_xform(prim_path)

    # STEP 5. Store the combined mesh
    DATA[prim_path]["position"] = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
    }
    DATA[prim_path]["orientation"] = {
        "x": float(quat[0]),
        "y": float(quat[1]),
        "z": float(quat[2]),
        "w": float(quat[3]),
    }

    return True


def apply(db: og.Database, **kwargs):
    for key, value in kwargs.items():
        if hasattr(db.outputs, key):
            setattr(db.outputs, key, value)


def setup(db: og.Database):
    # STEP 0. Get inputs
    prim_path = db.inputs.prim_path  # str
    if not prim_path:
        raise ValueError("prim_path input is empty.")

    downsample_voxel_size = db.inputs.downsample_voxel_size  # float
    if downsample_voxel_size is None or downsample_voxel_size <= 0:
        raise ValueError("downsample_voxel_size must be a positive float.")

    scale = db.inputs.scale  # float or (3,) array-like
    if scale is None:
        raise ValueError("scale input is None.")

    flag = db.inputs.flag  # bool
    if flag is None:
        raise ValueError("flag input is None.")

    if db.outputs.verts is None:
        raise ValueError("verts output is None.")
    if db.outputs.faces is None:
        raise ValueError("faces output is None.")

    x, y, z = db.outputs.x, db.outputs.y, db.outputs.z  # float
    if x is None or y is None or z is None:
        raise ValueError("x, y, z outputs must be valid floats.")

    qx, qy, qz, qw = (
        db.outputs.qx,
        db.outputs.qy,
        db.outputs.qz,
        db.outputs.qw,
    )  # float
    if qx is None or qy is None or qz is None or qw is None:
        raise ValueError("qx, qy, qz, qw outputs must be valid floats.")


def cleanup(db: og.Database):
    global DATA
    DATA = {}


def compute(db: og.Database):
    global DATA

    flag = db.inputs.flag  # bool

    verts = DATA.get(db.inputs.prim_path, {}).get("verts", [])
    faces = DATA.get(db.inputs.prim_path, {}).get("faces", [])

    position = DATA.get(db.inputs.prim_path, {}).get("position", {})
    orientation = DATA.get(db.inputs.prim_path, {}).get("orientation", {})

    x = position.get("x", 0.0)
    y = position.get("y", 0.0)
    z = position.get("z", 0.0)

    qx = orientation.get("x", 0.0)
    qy = orientation.get("y", 0.0)
    qz = orientation.get("z", 0.0)
    qw = orientation.get("w", 1.0)

    if (len(verts) == 0 or len(faces) == 0) or flag:
        get_mesh(db)

        print("Initialized combined mesh.")

        apply(
            db,
            verts=verts,
            faces=faces,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )

        return True

    else:
        # print("Updating pose only.")

        update_pose(db)

        apply(
            db,
            verts=verts,
            faces=faces,
            x=x,
            y=y,
            z=z,
            qx=qx,
            qy=qy,
            qz=qz,
            qw=qw,
        )

        return True
