"""Camera-independent pose consensus for multi-camera object tracking.

Transform convention:
    world_T_object = world_T_camera @ camera_T_object
All poses are 4x4 homogeneous matrices.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import yaml


@dataclass(frozen=True)
class ConsensusResult:
    available: bool
    consistent: bool
    should_reset: bool
    pose_world: Optional[np.ndarray]
    translation_error_m: float
    rotation_error_deg: float
    cameras_used: tuple
    reason: str


def _normalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized quaternion in xyzw order."""
    r = _normalize_rotation(rotation)
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s,
                      (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q = np.array([0.25 * s, (r[0, 1] + r[1, 0]) / s,
                          (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q = np.array([(r[0, 1] + r[1, 0]) / s, 0.25 * s,
                          (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q = np.array([(r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s,
                          (r[1, 0] - r[0, 1]) / s])
    q /= np.linalg.norm(q)
    return q


def _quaternion_to_rotation(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, dtype=np.float64)
    n = np.linalg.norm([x, y, z, w])
    if n == 0:
        raise ValueError("Quaternion must not be zero")
    x, y, z, w = np.array([x, y, z, w]) / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _average_quaternions(quaternions) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for q in quaternions:
        q = np.asarray(q, dtype=np.float64)
        q /= np.linalg.norm(q)
        accumulator += np.outer(q, q)
    _, vectors = np.linalg.eigh(accumulator)
    result = vectors[:, -1]
    if result[3] < 0:
        result = -result
    return result / np.linalg.norm(result)


def _rotation_distance_deg(a: np.ndarray, b: np.ndarray) -> float:
    relative = _normalize_rotation(a).T @ _normalize_rotation(b)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _axis_aligned_candidate(rotation: np.ndarray, aligned: np.ndarray,
                            axis_index: int, target: np.ndarray) -> np.ndarray:
    """Build the closest right-handed frame for one directed aligned axis."""
    secondary_index = (axis_index + 1) % 3
    secondary = rotation[:, secondary_index].copy()
    secondary -= aligned * np.dot(aligned, secondary)
    if np.linalg.norm(secondary) < 1e-9:
        secondary = target[:, secondary_index].copy()
        secondary -= aligned * np.dot(aligned, secondary)
    secondary /= np.linalg.norm(secondary)
    result = np.empty((3, 3), dtype=np.float64)
    result[:, axis_index] = aligned
    result[:, secondary_index] = secondary
    remaining_index = 3 - axis_index - secondary_index
    cyclic = (axis_index, secondary_index, remaining_index) in (
        (0, 1, 2), (1, 2, 0), (2, 0, 1)
    )
    result[:, remaining_index] = (np.cross(aligned, secondary) if cyclic
                                  else np.cross(secondary, aligned))
    return _normalize_rotation(result)


def _align_rotation_axis(rotation: np.ndarray, target: np.ndarray, axis: str,
                         soft: bool = False) -> np.ndarray:
    """Align an object axis while selecting the least disruptive full frame.

    In soft mode the configured axis is treated as an unoriented line. Both
    +target and -target are valid, and the candidate requiring the smallest
    geodesic rotation from the raw pose is selected. This automatically avoids
    a needless 180-degree flip of the other two axes on symmetric objects.
    """
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    rotation = _normalize_rotation(rotation)
    target = _normalize_rotation(target)
    target_axis = target[:, axis_index]
    candidates = [_axis_aligned_candidate(rotation, target_axis, axis_index, target)]
    if soft:
        candidates.append(
            _axis_aligned_candidate(rotation, -target_axis, axis_index, target)
        )
    return min(candidates, key=lambda candidate: _rotation_distance_deg(rotation, candidate))


def average_poses(poses) -> np.ndarray:
    poses = [np.asarray(pose, dtype=np.float64) for pose in poses]
    if not poses:
        raise ValueError("At least one pose is required")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    quaternions = [_rotation_to_quaternion(pose[:3, :3]) for pose in poses]
    result[:3, :3] = _quaternion_to_rotation(_average_quaternions(quaternions))
    return result


def _parse_transform(entry: Mapping) -> np.ndarray:
    if "matrix" in entry:
        matrix = np.asarray(entry["matrix"], dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Transform matrix must be 4x4")
        return matrix
    translation = entry.get("translation", {})
    rotation = entry.get("rotation_xyzw", entry.get("rotation", {}))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [translation.get("x", 0.0), translation.get("y", 0.0), translation.get("z", 0.0)]
    pose[:3, :3] = _quaternion_to_rotation([
        rotation.get("x", 0.0), rotation.get("y", 0.0),
        rotation.get("z", 0.0), rotation.get("w", 1.0)])
    return pose


def load_world_camera_transforms(path) -> tuple[str, Dict[str, np.ndarray]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    world_frame = str(config.get("world_frame", "world"))
    cameras = config.get("cameras", {})
    if not cameras:
        raise ValueError("Consensus config must contain at least one camera")
    transforms = {}
    for name, entry in cameras.items():
        transform = entry.get("world_T_camera", entry)
        transforms[str(name)] = _parse_transform(transform)
    return world_frame, transforms


class PoseConsensus:
    """Compare camera estimates in one frame and fuse them when consistent."""
    def __init__(self, world_T_camera: Mapping[str, np.ndarray],
                 translation_tolerance_m: float = 0.05,
                 rotation_tolerance_deg: float = 10.0,
                 min_cameras: int = 2,
                 failures_before_reset: int = 3,
                 force_align_axis: Optional[str] = None,
                 soft_force_align: bool = False):
        if min_cameras < 2:
            raise ValueError("min_cameras must be at least 2")
        if failures_before_reset < 1:
            raise ValueError("failures_before_reset must be positive")
        self.world_T_camera = {str(k): np.asarray(v, dtype=np.float64) for k, v in world_T_camera.items()}
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.min_cameras = int(min_cameras)
        self.failures_before_reset = int(failures_before_reset)
        if force_align_axis is not None:
            force_align_axis = str(force_align_axis).lower()
            if force_align_axis not in {"x", "y", "z"}:
                raise ValueError("force_align_axis must be one of: x, y, z")
        self.force_align_axis = force_align_axis
        self.soft_force_align = bool(soft_force_align)
        self._failure_streaks: Dict[str, int] = {}

    @classmethod
    def from_yaml(cls, path, **overrides):
        world_frame, transforms = load_world_camera_transforms(path)
        with Path(path).open("r", encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream) or {}
        settings = dict(cfg.get("consensus", {}))
        settings.update({k: v for k, v in overrides.items() if v is not None})
        instance = cls(transforms, **settings)
        instance.world_frame = world_frame
        return instance

    def reset_state(self, object_name: Optional[str] = None):
        if object_name is None:
            self._failure_streaks.clear()
        else:
            self._failure_streaks.pop(object_name, None)

    def align_camera_pose_for_display(self, camera: str, camera_T_object: np.ndarray,
                                      pose_world: np.ndarray) -> np.ndarray:
        """Return a camera-frame pose whose selected axis matches fused world pose.

        Translation is intentionally retained from the raw tracker pose so the
        overlay remains attached to the observed object. Only the configured
        axis direction is canonicalized in the common world frame.
        """
        pose = np.asarray(camera_T_object, dtype=np.float64).copy()
        if self.force_align_axis is None or camera not in self.world_T_camera:
            return pose
        world_T_object = self.world_T_camera[camera] @ pose
        world_T_object[:3, :3] = _align_rotation_axis(
            world_T_object[:3, :3], pose_world[:3, :3], self.force_align_axis,
            soft=self.soft_force_align,
        )
        aligned_camera_pose = np.linalg.inv(self.world_T_camera[camera]) @ world_T_object
        aligned_camera_pose[:3, 3] = pose[:3, 3]
        return aligned_camera_pose

    def evaluate(self, object_name: str,
                 camera_T_object: Mapping[str, Optional[np.ndarray]]) -> ConsensusResult:
        world_poses = {}
        for camera, object_pose in camera_T_object.items():
            if object_pose is None or camera not in self.world_T_camera:
                continue
            pose = np.asarray(object_pose, dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                continue
            world_poses[camera] = self.world_T_camera[camera] @ pose

        cameras = tuple(sorted(world_poses))
        if len(cameras) < self.min_cameras:
            self._failure_streaks[object_name] = 0
            return ConsensusResult(False, False, False, None, 0.0, 0.0, cameras,
                                   f"need {self.min_cameras} cameras, got {len(cameras)}")

        max_translation = 0.0
        max_rotation = 0.0
        for i, first in enumerate(cameras):
            for second in cameras[i + 1:]:
                a, b = world_poses[first], world_poses[second]
                max_translation = max(max_translation, float(np.linalg.norm(a[:3, 3] - b[:3, 3])))
                max_rotation = max(max_rotation, _rotation_distance_deg(a[:3, :3], b[:3, :3]))

        consistent = (max_translation <= self.translation_tolerance_m and
                      max_rotation <= self.rotation_tolerance_deg)
        if consistent:
            self._failure_streaks[object_name] = 0
            fused = average_poses([world_poses[camera] for camera in cameras])
            if self.force_align_axis is not None:
                fused[:3, :3] = _align_rotation_axis(
                    fused[:3, :3], world_poses[cameras[0]][:3, :3],
                    self.force_align_axis, soft=self.soft_force_align,
                )
            reason = "camera poses agree"
        else:
            streak = self._failure_streaks.get(object_name, 0) + 1
            self._failure_streaks[object_name] = streak
            fused = None
            reason = f"disagreement streak {streak}/{self.failures_before_reset}"
        should_reset = (not consistent and
                        self._failure_streaks[object_name] >= self.failures_before_reset)
        if should_reset:
            self._failure_streaks[object_name] = 0
        return ConsensusResult(True, consistent, should_reset, fused,
                               max_translation, max_rotation, cameras, reason)
