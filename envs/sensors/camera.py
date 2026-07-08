from isaaclab.sensors import TiledCameraCfg, TiledCamera
from isaaclab.utils import configclass

import torch
import torchvision.transforms.functional as F
from numbers import Real
import re
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc import UipcInteractiveScene

@configclass
class CameraCfg(TiledCameraCfg):
    name: str = 'camera'

class CameraManager:
    def __init__(self, cfg_list: list[CameraCfg], task:'BaseTask'):
        self.scene = task.scene
        self.cfg_list = cfg_list
        self.cameras = {}

    def setup(self): 
        self.cameras = {
            cam_cfg.name: self.add_camera(cam_cfg) for cam_cfg in self.cfg_list
        }

    def _patch_scalar_scale_reader(self):
        try:
            from isaaclab.sim.views.xform_prim_view import XformPrimView
            from pxr import Gf, Vt
        except Exception as exc:
            print(f"Warning: cannot patch IsaacLab scalar scale reader: {exc}")
            return

        if getattr(XformPrimView, "_univtac_scalar_scale_patch", False):
            return

        def _get_scales_usd_compat(view_self, indices=None):
            if indices is None or indices == slice(None):
                indices_list = view_self._ALL_INDICES
            else:
                indices_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)

            scales = Vt.Vec3dArray(len(indices_list))
            for idx, prim_idx in enumerate(indices_list):
                prim = view_self._prims[prim_idx]
                attr = prim.GetAttribute("xformOp:scale")
                value = attr.Get() if attr and attr.HasValue() else None
                if value is None:
                    value = Gf.Vec3d(1.0, 1.0, 1.0)
                elif isinstance(value, Real):
                    value = Gf.Vec3d(float(value), float(value), float(value))
                else:
                    try:
                        if len(value) == 3:
                            value = Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))
                        else:
                            scalar = float(value)
                            value = Gf.Vec3d(scalar, scalar, scalar)
                    except TypeError:
                        scalar = float(value)
                        value = Gf.Vec3d(scalar, scalar, scalar)
                scales[idx] = value

            return torch.tensor(np.array(scales), dtype=torch.float32, device=view_self._device)

        XformPrimView._get_scales_usd = _get_scales_usd_compat
        XformPrimView._univtac_scalar_scale_patch = True
        print("Patched IsaacLab XformPrimView scalar scale reader for UniVTAC cameras")

    def _normalize_scalar_scale(self, cam_cfg: CameraCfg):
        try:
            import isaaclab.sim as sim_utils
            from pxr import Gf, Sdf
            import omni.usd
        except Exception as exc:
            print(f"Warning: cannot import USD scale helpers for camera {cam_cfg.name}: {exc}")
            return

        prims = []
        seen = set()

        def add_prim(prim):
            if prim and prim.IsValid():
                path = str(prim.GetPath())
                if path not in seen:
                    prims.append(prim)
                    seen.add(path)

        for prim in sim_utils.find_matching_prims(cam_cfg.prim_path):
            add_prim(prim)

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            patterns = [cam_cfg.prim_path]
            if cam_cfg.prim_path.endswith("/Camera"):
                patterns.append(cam_cfg.prim_path[:-len("/Camera")])
            compiled = [re.compile(f"^{pattern}$") for pattern in patterns]
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if any(pattern.match(path) for pattern in compiled):
                    add_prim(prim)

        for prim in prims:
            attr = prim.GetAttribute("xformOp:scale")
            if not attr or not attr.HasValue():
                continue
            value = attr.Get()
            if not isinstance(value, Real):
                continue

            scale = Gf.Vec3d(float(value), float(value), float(value))
            try:
                attr.Set(scale)
            except Exception:
                prim.RemoveProperty("xformOp:scale")
                attr = prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False)
                attr.Set(scale)
            print(f"Normalized scalar camera scale for {prim.GetPath()} to {scale}")

    def add_camera(self, cam_cfg: CameraCfg):
        self._patch_scalar_scale_reader()
        self._normalize_scalar_scale(cam_cfg)
        camera = TiledCamera(cam_cfg)
        camera._initialize_impl()
        camera._is_initialized = True
        self.scene.sensors[f'camera_{cam_cfg.name}'] = camera
        return camera
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgba']
        for name, cam in self.cameras.items():
            obs[name] = {}
            for data_type in data_types:
                if data_type == 'rgb':
                    obs[name]['rgb'] = cam.data.output['rgb'].squeeze(0)
                elif data_type == 'rgba':
                    obs[name]['rgba'] = cam.data.output['rgba'].squeeze(0)
                elif data_type == 'depth':
                    obs[name]['depth'] = cam.data.output['depth'].squeeze(0)
        return obs
