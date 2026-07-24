"""Shared multi-camera capture and batched FoundationPose orchestration."""
from dataclasses import dataclass
import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from utils.tracking_utils import (
    build_estimator, create_shared_estimator_components, get_sam3_mask,
    intrinsics_to_K, load_camera_config, load_mesh,
)


@dataclass
class TrackingUnit:
    object_idx: int
    camera_idx: int
    obj: Dict[str, Any]
    frame: Dict[str, Any]


class MultiCameraCapture:
    def __init__(self, camera_names, width=1280, height=720, fps=30):
        import pyrealsense2 as rs
        self.rs = rs
        self.camera_names = list(camera_names)
        self.entries = []
        for name in self.camera_names:
            camera = load_camera_config(name)
            pipeline, config = rs.pipeline(), rs.config()
            if camera.get("serial"):
                config.enable_device(str(camera["serial"]))
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            profile = pipeline.start(config)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.entries.append({
                "name": name, "frame_id": camera["frame_id"], "pipeline": pipeline,
                "align": rs.align(rs.stream.color), "K": intrinsics_to_K(intr),
                "depth_scale": profile.get_device().first_depth_sensor().get_depth_scale(),
            })
            logging.info("Started camera '%s'", name)

    def read(self):
        frames = []
        for entry in self.entries:
            fs = entry["align"].process(entry["pipeline"].wait_for_frames())
            depth_frame, color_frame = fs.get_depth_frame(), fs.get_color_frame()
            if not depth_frame or not color_frame:
                return None
            bgr = np.asanyarray(color_frame.get_data())
            frames.append({
                "name": entry["name"], "frame_id": entry["frame_id"], "K": entry["K"],
                "color_bgr": bgr, "color_rgb": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                "depth": np.asanyarray(depth_frame.get_data()).astype(np.float32) * entry["depth_scale"],
            })
        return frames

    def stop(self):
        for entry in self.entries:
            try: entry["pipeline"].stop()
            except Exception: logging.exception("Could not stop camera '%s'", entry["name"])


def build_multi_camera_state(object_names, camera_names, debug=0, mesh_origin=False):
    scorer, refiner, glctx = create_shared_estimator_components()
    tracked = []
    colors = [(0,255,0),(0,0,255),(255,0,0),(0,165,255),(255,255,0)]
    for i, name in enumerate(object_names):
        mesh_path, _ = load_mesh(name)
        debug_dir = f"/tmp/fp_debug/{name}"
        os.makedirs(debug_dir, exist_ok=True)
        est, mesh, to_origin, bbox = build_estimator(
            mesh_path, debug_dir=debug_dir, debug=debug,
            scorer=scorer, refiner=refiner, glctx=glctx)
        tracked.append({"name": name, "est": est, "mesh": mesh, "to_origin": to_origin,
                        "bbox": bbox, "color_bgr": colors[i % len(colors)],
                        "mesh_origin": mesh_origin,
                        "camera_states": [{"initialized": False, "pose": None, "pose_last": None}
                                          for _ in camera_names]})
    return tracked, scorer, refiner, glctx


def register_all_cameras(tracked, frames, sam_processor, iteration):
    for camera_idx, frame in enumerate(frames):
        for obj in tracked:
            state = obj["camera_states"][camera_idx]
            if state["initialized"]:
                continue
            mask = get_sam3_mask(sam_processor, frame["color_rgb"], obj["name"])
            if mask is None or mask.sum() <= 100:
                continue
            if ((frame["depth"] >= .001) & mask.astype(bool)).sum() < 4:
                logging.warning("Insufficient depth for %s on %s", obj["name"], frame["name"])
                continue
            # register mutates estimator.pose_last; immediately move it into camera state
            pose = obj["est"].register(K=frame["K"], rgb=frame["color_rgb"],
                                       depth=frame["depth"], ob_mask=mask.astype(bool),
                                       iteration=iteration)
            state.update(initialized=True, pose=pose, pose_last=obj["est"].pose_last.detach().clone())
            obj["est"].pose_last = None
            logging.info("Registered '%s' on '%s'", obj["name"], frame["name"])


def collect_tracking_units(tracked, frames):
    return [TrackingUnit(oi, ci, obj, frames[ci])
            for oi, obj in enumerate(tracked)
            for ci, state in enumerate(obj["camera_states"])
            if state["initialized"]]


def track_batch(refiner, glctx, units, refine_iter):
    if not units:
        return
    items = []
    for unit in units:
        est, frame = unit.obj["est"], unit.frame
        state = unit.obj["camera_states"][unit.camera_idx]
        depth = torch.as_tensor(frame["depth"], device="cuda", dtype=torch.float)
        from Utils import erode_depth, bilateral_filter_depth, depth2xyzmap_batch
        depth = bilateral_filter_depth(erode_depth(depth, radius=2, device="cuda"), radius=2, device="cuda")
        xyz = depth2xyzmap_batch(depth[None], torch.as_tensor(frame["K"], device="cuda", dtype=torch.float)[None], zfar=np.inf)[0]
        items.append({"rgb": frame["color_rgb"], "depth": depth, "K": frame["K"],
                      "ob_in_cams": state["pose_last"].reshape(1,4,4).detach().cpu().numpy(),
                      "xyz_map": xyz, "normal_map": None, "mesh": est.mesh,
                      "mesh_tensors": est.mesh_tensors, "mesh_diameter": est.diameter, "glctx": glctx})
    poses = refiner.predict_multi(items, iteration=refine_iter)
    for unit, centered in zip(units, poses):
        state, est = unit.obj["camera_states"][unit.camera_idx], unit.obj["est"]
        state["pose_last"] = centered.detach().clone()
        state["pose"] = (centered @ est.get_tf_to_centered_mesh()).detach().cpu().numpy().reshape(4,4)
