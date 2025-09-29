# USD -> CollisionObject (ROS2 Humble, Python)
# 의존성: pip install usd-core numpy
from typing import List, Tuple, Optional
import os
import numpy as np

from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import Mesh, MeshTriangle
from moveit_msgs.msg import CollisionObject
from ament_index_python.packages import get_package_share_directory


# USD
from pxr import Usd, UsdGeom, Gf, UsdPhysics

# ---------- 공용: package:// 경로 처리 ----------
def _resolve_resource_path(url_or_path: str) -> str:
    if url_or_path.startswith("package://"):
        rest = url_or_path[len("package://"):]
        parts = rest.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid package URL: {url_or_path}")
        pkg, subpath = parts
        base = get_package_share_directory(pkg)
        return os.path.join(base, subpath)
    return url_or_path

# ---------- 공용: shape_msgs/Mesh 빌더 ----------
def _build_shape_msgs_mesh(vertices: np.ndarray, faces_tri: np.ndarray) -> Mesh:
    msg = Mesh()
    for vx, vy, vz in vertices:
        msg.vertices.append(Point(x=float(vx), y=float(vy), z=float(vz)))
    for i0, i1, i2 in faces_tri.astype(np.int32):
        tri = MeshTriangle(vertex_indices=[int(i0), int(i1), int(i2)])
        msg.triangles.append(tri)
    return msg

# ---------- USD: 폴리곤 → 삼각형 인덱스 변환 (팬-트라이앵글 방식) ----------
def _triangulate_counts_indices(face_counts: List[int], face_indices: List[int]) -> np.ndarray:
    """
    USD Mesh는 faceVertexCounts/Indices로 폴리곤을 정의.
    이를 (M,3) 정수 배열의 삼각형 인덱스로 변환.
    """
    tris = []
    idx_cursor = 0
    for n in face_counts:
        inds = face_indices[idx_cursor: idx_cursor + n]
        idx_cursor += n
        # n==3이면 그대로, n>3이면 v0,vk,vk+1 팬 방식
        if n == 3:
            tris.append([inds[0], inds[1], inds[2]])
        elif n > 3:
            for k in range(1, n - 1):
                tris.append([inds[0], inds[k], inds[k + 1]])
        # n<3은 무시(비정상 데이터)
    if len(tris) == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.array(tris, dtype=np.int32)

# ---------- USD: 스테이지 단위(미터 단위) 스케일 얻기 ----------
def _usd_stage_meters_per_unit(stage: Usd.Stage) -> float:
    """
    USD 기본 단위는 종종 cm(0.01m). Stage metadata의 metersPerUnit 사용.
    없으면 0.01(=cm)로 가정하는 프로젝트가 많지만, 여기선 스펙상 기본 1.0로 둡니다.
    필요시 정책에 맞게 기본값을 바꾸세요.
    """
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if mpu is None or mpu <= 0:
        # 프로젝트 정책에 맞게 결정: 대부분의 DCC는 0.01을 쓰는 편.
        # 안전하게 1.0을 기본으로 두고, 필요시 호출자가 scale로 보정하세요.
        return 1.0
    return float(mpu)

# ---------- USD: 메시들을 읽어서 (verts, faces) 리스트 반환 ----------
def _read_usd_meshes_baked(path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    스테이지의 모든 UsdGeom.Mesh를 찾고,
    월드 변환을 버텍스에 '구워서(bake)' 적용한 뒤,
    폴리곤을 삼각형 인덱스로 변환하여 반환.

    Returns:
      verts_list: [ (Ni,3) float32 ... ]
      faces_list: [ (Mi,3) int32   ... ]
    """
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {path}")

    mpu = _usd_stage_meters_per_unit(stage)  # metersPerUnit
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    verts_list = []
    faces_list = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)

        # Points (버텍스) & Face topology
        points = mesh.GetPointsAttr().Get()  # List[Gf.Vec3f/d]
        if points is None or len(points) == 0:
            continue
        face_counts = mesh.GetFaceVertexCountsAttr().Get() or []
        face_indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if len(face_counts) == 0 or len(face_indices) == 0:
            continue

        # numpy 변환 + USD 단위 → 미터 스케일 적용
        P = np.array([[p[0], p[1], p[2]] for p in points], dtype=np.float64) * mpu

        # 월드 변환 행렬 얻기
        M: Gf.Matrix4d = xform_cache.GetLocalToWorld(prim)
        # 3D 점에 4x4 적용 (동차좌표)
        Pw = (M.Transform(P)).astype(np.float64)  # pxr는 Vec3 배열에 일괄 적용 가능

        # 폴리곤 → 삼각형 인덱스
        Ftri = _triangulate_counts_indices(face_counts, face_indices)

        verts_list.append(Pw.astype(np.float64))
        faces_list.append(Ftri.astype(np.int32))

    return verts_list, faces_list

# ---------- 공개 함수: USD → CollisionObject ----------
def load_usd_as_collision_object(
    url_or_path: str,
    *,
    object_id: str,
    frame_id: str = "world",
    pose: Optional[Pose] = None,
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    combine_submeshes: bool = True,   # True면 모든 메시를 하나로 합쳐 1개의 Mesh로 넣음
    bake_xforms: bool = True          # True면 월드 변환을 버텍스에 적용하고 pose는 그대로 사용(기본 identity 권장)
) -> CollisionObject:
    """
    USD(USDA/USDZ 포함)에서 메시(들)를 읽어 moveit_msgs/CollisionObject로 반환.

    - 모든 UsdGeom.Mesh를 탐색
    - faceVertexCounts/Indices를 삼각화
    - metersPerUnit 반영
    - (옵션) 월드 변환을 버텍스에 베이크
    - (옵션) 여러 메시를 하나로 합쳐 단일 Mesh로 생성

    Args:
      url_or_path : 파일 경로 또는 package:// 경로
      object_id   : PlanningScene에서 고유 ID
      frame_id    : Pose가 표현되는 TF 프레임
      pose        : CollisionObject의 포즈(기본: 원점 단위쿼터니언)
      scale       : 최종 버텍스에 추가 스케일 (단위 보정/크기 조정)
      combine_submeshes : True면 모든 USD Mesh를 합쳐 1개의 Mesh로 만듭니다
      bake_xforms : True면 USD Xform을 버텍스에 적용(권장); False면 적용 안 함

    Returns:
      moveit_msgs.msg.CollisionObject
    """
    path = _resolve_resource_path(url_or_path)
    verts_list, faces_list = _read_usd_meshes_baked(path)

    if len(verts_list) == 0:
        raise ValueError(f"No UsdGeom.Mesh found in: {path}")

    # 추가 스케일 적용 (예: mm->m 보정 등)
    S = np.asarray(scale, dtype=np.float64).reshape(1, 3)
    verts_list = [v.astype(np.float64) * S for v in verts_list]

    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = frame_id

    # Pose 기본값
    if pose is None:
        pose = Pose(
            position=Point(0.0, 0.0, 0.0),
            orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
        )

    if combine_submeshes:
        # 인덱스 오프셋을 관리하며 하나로 병합
        all_verts = []
        all_faces = []
        offset = 0
        for v, f in zip(verts_list, faces_list):
            all_verts.append(v)
            if f.size > 0:
                all_faces.append(f + offset)
            offset += v.shape[0]
        V = np.vstack(all_verts).astype(np.float64)
        F = (np.vstack(all_faces).astype(np.int32)) if len(all_faces) else np.zeros((0, 3), dtype=np.int32)

        mesh_msg = _build_shape_msgs_mesh(V, F)
        co.meshes.append(mesh_msg)
        co.mesh_poses.append(pose)
    else:
        # 서브메시를 각각 별도 Mesh로 등록
        for v, f in zip(verts_list, faces_list):
            mesh_msg = _build_shape_msgs_mesh(v, f)
            co.meshes.append(mesh_msg)
            co.mesh_poses.append(pose)

    co.operation = CollisionObject.ADD
    return co


# 단일 CollisionObject로 합쳐서 추가
co = load_usd_as_collision_object(
    "/home/min/7cmdehdrb/ggg/Collected_gg/omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/Robots/Robotiq/2F-140/2f140_instanceable.usd",
    object_id="fixture_usd",
    frame_id="world",
    combine_submeshes=True,   # 모든 서브메시 → 1개 Mesh
    scale=(1.0, 1.0, 1.0)
)

print(co)