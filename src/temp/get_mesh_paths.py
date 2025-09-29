import omni.usd
from pxr import Usd
import numpy as np
import carb


class MeshFinder(object):

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


mesh_paths = []


def setup(db: og.Database):
    global mesh_paths

    # 노드 초기화 시점에서 한번 실행
    print("[ScriptNode] setup called.")

    prim_path = db.inputs.prim_path
    if not prim_path:
        carb.log_warn("prim_path input is empty.")
        return False

    mesh_paths = MeshFinder.find_mesh_paths(prim_path)
    print(f"Found {len(mesh_paths)} mesh paths under {prim_path}.")
    return True


def cleanup(db: og.Database):
    global mesh_paths
    mesh_paths = []


def compute(db: og.Database):
    global mesh_paths
    db.outputs.mesh_paths = mesh_paths

    print(f"Output {len(mesh_paths)} mesh paths.")

    return True
