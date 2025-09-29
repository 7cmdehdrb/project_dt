# Standard library
import os
import csv
from typing import Tuple, Optional

# Third-party
import numpy as np
import trimesh

# ROS2 core
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, qos_profile_system_default

# ROS2 messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from moveit_msgs.msg import *
from shape_msgs.msg import *

# TF
from tf2_ros import *

# ROS2 MoveIt
from moveit2_commander import *
from rotutils import *

# ROS2 package index
from ament_index_python.packages import get_package_share_directory


import numpy as np


def apply_pose(verts, pose4x4):
    # verts: (N,3), pose: (4,4)
    N = verts.shape[0]
    homo = np.c_[verts, np.ones((N, 1), dtype=verts.dtype)]
    out = (homo @ pose4x4.T)[:, :3]
    return out


def merge_meshes_with_poses(meshes, poses):
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
        v_w = apply_pose(v, T)
        all_verts.append(v_w)
        all_faces.append(f + offset)
        offset += v.shape[0]

    merged_verts = np.vstack(all_verts).astype(np.float64, copy=False)
    merged_faces = np.vstack(all_faces).astype(np.int32, copy=False)
    return merged_verts, merged_faces


def weld_vertices(verts, faces, tol=1e-6):
    """
    좌표가 tol 이내로 같은 정점을 하나로 합침.
    간단히 좌표를 라운딩해서 unique 수행(빠르고 실용적).
    """
    if verts.size == 0:
        return verts, faces

    key = np.round(verts / tol).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    new_verts = np.zeros((uniq.shape[0], 3), dtype=verts.dtype)
    # 각 클러스터 평균으로 대표
    for i in range(uniq.shape[0]):
        new_verts[i] = verts[inv == i].mean(axis=0)

    new_faces = inv[faces]
    # 퇴화 면 제거
    mask = (
        (new_faces[:, 0] != new_faces[:, 1])
        & (new_faces[:, 1] != new_faces[:, 2])
        & (new_faces[:, 2] != new_faces[:, 0])
    )
    new_faces = new_faces[mask]
    return new_verts, new_faces


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


def _resolve_resource_path(url_or_path: str) -> str:
    """
    Supports:
      - Absolute/relative filesystem paths
      - package://<pkg_name>/<subpath> (ROS package resource)
    """
    if url_or_path.startswith("package://"):
        # package://pkg_name/sub/dir/model.stl
        rest = url_or_path[len("package://") :]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid package URL: {url_or_path}")
        pkg, subpath = parts
        base = get_package_share_directory(pkg)
        return os.path.join(base, subpath)
    return url_or_path


def _from_csv(vert_file: str, face_file: str) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []

    with open(vert_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Read vertex
            vx = float(row["verts_x"])
            vy = float(row["verts_y"])
            vz = float(row["verts_z"])
            verts.append([vx, vy, vz])

    with open(face_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Read face indices
            i0 = int(row["faces_v1"])
            i1 = int(row["faces_v2"])
            i2 = int(row["faces_v3"])
            faces.append([i0, i1, i2])

    verts_np = np.array(verts, dtype=np.float64)
    faces_np = np.array(faces, dtype=np.int32)
    return verts_np, faces_np


def _as_shape_msgs_mesh(tm: trimesh.Trimesh, scale: Tuple[float, float, float]) -> Mesh:
    """
    Convert a trimesh.Trimesh to shape_msgs/Mesh with optional per-axis scale.
    Ensures triangles only.
    """
    # Force triangulation if faces have more than 3 verts
    if tm.faces.shape[1] != 3:
        tm = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=True)

    # Apply scale
    S = np.asarray(scale, dtype=np.float64).reshape(1, 3)
    verts = np.asarray(tm.vertices, dtype=np.float64) * S
    faces = np.asarray(tm.faces, dtype=np.int32)

    print(verts.shape, faces.shape)

    msg = Mesh()
    for vx, vy, vz in verts:
        msg.vertices.append(Point(x=float(vx), y=float(vy), z=float(vz)))
    for i0, i1, i2 in faces:
        tri = MeshTriangle(vertex_indices=[int(i0), int(i1), int(i2)])
        msg.triangles.append(tri)

    return msg


def _as_shape_msgs_mesh_from_csv(verts: np.ndarray, faces: np.ndarray) -> Mesh:
    """
    Convert a trimesh.Trimesh to shape_msgs/Mesh with optional per-axis scale.
    Ensures triangles only.
    """
    print(verts.shape, faces.shape)

    msg = Mesh()
    for vx, vy, vz in verts:
        msg.vertices.append(Point(x=float(vx), y=float(vy), z=float(vz)))
    for i0, i1, i2 in faces:
        tri = MeshTriangle(vertex_indices=[int(i0), int(i1), int(i2)])
        msg.triangles.append(tri)

    return msg


def load_mesh_as_collision_object(
    url_or_path: str,
    *,
    object_id: str,
    frame_id: str = "world",
    pose: Optional[Pose] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CollisionObject:
    """
    Read an STL/OBJ (and most formats trimesh supports) and return a moveit_msgs/CollisionObject.

    Args:
      url_or_path: filesystem path or 'package://<pkg>/<subpath>'
      object_id: unique ID for the object in the planning scene
      frame_id: TF frame in which the pose is expressed
      pose: geometry_msgs/Pose of the mesh origin; if None, identity
      scale: per-axis scale applied to mesh vertices

    Returns:
      moveit_msgs.msg.CollisionObject with meshes[0] and mesh_poses[0] set, operation=ADD
    """
    path = _resolve_resource_path(url_or_path)

    # Load with trimesh (force mesh; will merge if the file has multiple sub-meshes)
    tm = trimesh.load(path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        # If it's a Scene, combine into a single mesh in scene coordinates
        tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
    if not isinstance(tm, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from file: {path}")

    mesh_msg = _as_shape_msgs_mesh(tm, scale)

    # Default pose = identity
    if pose is None:
        pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = frame_id
    co.meshes.append(mesh_msg)
    co.mesh_poses.append(pose)
    co.operation = CollisionObject.ADD
    return co


def load_mesh_as_collision_object_from_csv(
    vert_csv_path: str,
    face_csv_path: str,
    *,
    object_id: str,
    frame_id: str = "world",
    pose: Optional[Pose] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> CollisionObject:
    verts, faces = _from_csv(vert_csv_path, face_csv_path)
    new_verts, new_faces = downsample_vertex_clustering(
        verts, faces, voxel_size=0.1, reducer="mean"
    )

    L = 10
    # Sample 4x4 transform matrices (translation, rotation)
    poses = [
        np.array(
            [
                [1, 0, 0, np.random.uniform(-1.0, 1.0)],
                [0, 1, 0, np.random.uniform(-1.0, 1.0)],
                [0, 0, 1, np.random.uniform(-1.0, 1.0)],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        for _ in range(L)
    ]
    meshes = [(new_verts, new_faces) for _ in poses]
    V_merged, F_merged = merge_meshes_with_poses(meshes=meshes, poses=poses)

    # mesh_msg = _as_shape_msgs_mesh_from_csv(new_verts, new_faces)
    mesh_msg = _as_shape_msgs_mesh_from_csv(V_merged, F_merged)

    # Default pose = identity
    if pose is None:
        pose = Pose(
            position=Point(x=0.0, y=0.0, z=0.0),
            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        )

    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = frame_id
    co.meshes.append(mesh_msg)
    co.mesh_poses.append(pose)
    co.operation = CollisionObject.ADD
    return co


def make_simple_box() -> CollisionObject:
    # 박스 primitive 생성 (x=0.1, y=0.1, z=0.1 m)
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [0.1, 0.1, 0.1]

    # 박스의 pose (world 원점, 회전 없음)
    pose = Pose(
        position=Point(x=1.0, y=1.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )

    # CollisionObject 구성
    co = CollisionObject()
    co.id = "simple_box"
    co.header.frame_id = "world"
    co.primitives.append(box)
    co.primitive_poses.append(pose)
    co.operation = CollisionObject.ADD

    return co


class TestNode(Node):
    def __init__(self):
        super().__init__("mesh_test_node")

        self._get_planning_scene_manager = GetPlanningScene_ServiceManager(node=self)
        self._apply_planning_scene_manager = ApplyPlanningScene_ServiceManager(
            node=self
        )

        self._co = load_mesh_as_collision_object(
            "/home/min/7cmdehdrb/project_th/Buckle.stl",
            object_id="fixture_mesh",
            frame_id="world",
            pose=Pose(
                position=Point(x=-0.5, y=-0.5, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=(0.001, 0.001, 0.001),
        )
        self._csv_co = load_mesh_as_collision_object_from_csv(
            vert_csv_path="/home/min/7cmdehdrb/project_th/verts.csv",
            face_csv_path="/home/min/7cmdehdrb/project_th/faces.csv",
            object_id="fixture_csv",
            frame_id="world",
            pose=Pose(
                position=Point(x=-0.5, y=0.5, z=0.0),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            scale=(1.0, 1.0, 1.0),
        )
        self._test_co = make_simple_box()

    def update_planning_scene(self):
        self._planning_scene = self._get_planning_scene_manager.run()

        self._apply_planning_scene_manager.run(
            collision_objects=[self._csv_co],
            scene=self._planning_scene,
        )

    @property
    def collision_object(self) -> CollisionObject:
        return self._co


def main(args=None):
    rclpy.init(args=args)

    import threading

    node = TestNode()

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    rate = node.create_rate(5.0)
    while rclpy.ok():
        node.update_planning_scene()
        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
