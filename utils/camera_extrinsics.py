"""Relative camera extrinsic calibration from observations of a shared rigid probe.

Pose convention:
    reference_T_camera_i = reference_T_probe @ inverse(camera_i_T_probe)

Every accepted sample must contain a simultaneous camera_T_probe measurement for
all configured cameras. The reference camera transform is fixed to identity,
which removes the global gauge freedom.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def invert_pose(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = pose[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ pose[:3, 3]
    return result


def pose_to_vector(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec()]


def vector_to_pose(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    pose = np.eye(4)
    pose[:3, 3] = vector[:3]
    pose[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    return pose


def rotation_error(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation-vector residual taking b to a."""
    return Rotation.from_matrix(a @ b.T).as_rotvec()


def average_poses(poses: Iterable[np.ndarray]) -> np.ndarray:
    poses = [np.asarray(p, dtype=np.float64) for p in poses]
    if not poses:
        raise ValueError('At least one pose is required')
    result = np.eye(4)
    result[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    rotations = Rotation.from_matrix([p[:3, :3] for p in poses])
    result[:3, :3] = rotations.mean().as_matrix()
    return result


@dataclass(frozen=True)
class ExtrinsicCalibrationResult:
    reference_camera: str
    reference_T_camera: Dict[str, np.ndarray]
    sample_count: int
    translation_rmse_m: Dict[str, float]
    rotation_rmse_deg: Dict[str, float]
    optimizer_cost: float
    optimizer_success: bool
    optimizer_message: str


class RelativeCameraCalibrator:
    """Collect probe poses and solve camera transforms with robust least squares."""
    def __init__(self, camera_names, reference_camera, sample_count=100,
                 min_probe_translation_m=0.01, min_probe_rotation_deg=3.0,
                 translation_scale_m=0.02, rotation_scale_deg=3.0,
                 robust_loss='soft_l1'):
        self.camera_names = tuple(camera_names)
        if reference_camera not in self.camera_names:
            raise ValueError('reference_camera must be in camera_names')
        if len(self.camera_names) < 2:
            raise ValueError('At least two cameras are required')
        self.reference_camera = reference_camera
        self.sample_count = int(sample_count)
        self.min_probe_translation_m = float(min_probe_translation_m)
        self.min_probe_rotation_deg = float(min_probe_rotation_deg)
        self.translation_scale_m = float(translation_scale_m)
        self.rotation_scale_rad = np.deg2rad(float(rotation_scale_deg))
        self.robust_loss = robust_loss
        self.samples = []

    @property
    def complete(self):
        return len(self.samples) >= self.sample_count

    def add_sample(self, camera_T_probe: Mapping[str, np.ndarray]) -> bool:
        if self.complete or any(name not in camera_T_probe for name in self.camera_names):
            return False
        sample = {}
        for name in self.camera_names:
            pose = np.asarray(camera_T_probe[name], dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                return False
            sample[name] = pose.copy()
        if self.samples:
            old = self.samples[-1][self.reference_camera]
            new = sample[self.reference_camera]
            distance = np.linalg.norm(old[:3, 3] - new[:3, 3])
            angle = np.degrees(np.linalg.norm(rotation_error(new[:3, :3], old[:3, :3])))
            if distance < self.min_probe_translation_m and angle < self.min_probe_rotation_deg:
                return False
        self.samples.append(sample)
        return True

    def _initial_transform(self, camera):
        candidates = [s[self.reference_camera] @ invert_pose(s[camera]) for s in self.samples]
        return average_poses(candidates)

    def solve(self) -> ExtrinsicCalibrationResult:
        if len(self.samples) < 3:
            raise RuntimeError('At least three accepted samples are required')
        others = [c for c in self.camera_names if c != self.reference_camera]
        initial = np.concatenate([pose_to_vector(self._initial_transform(c)) for c in others])

        def unpack(x):
            values = {self.reference_camera: np.eye(4)}
            for i, camera in enumerate(others):
                values[camera] = vector_to_pose(x[6*i:6*(i+1)])
            return values

        def residuals(x):
            extrinsics = unpack(x)
            residual = []
            for sample in self.samples:
                reference_T_probe = sample[self.reference_camera]
                for camera in others:
                    predicted = extrinsics[camera] @ sample[camera]
                    residual.extend((reference_T_probe[:3, 3] - predicted[:3, 3]) /
                                    self.translation_scale_m)
                    residual.extend(rotation_error(reference_T_probe[:3, :3],
                                                   predicted[:3, :3]) /
                                    self.rotation_scale_rad)
            return np.asarray(residual)

        optimum = least_squares(residuals, initial, loss=self.robust_loss,
                                f_scale=1.0, max_nfev=300)
        extrinsics = unpack(optimum.x)
        trans_rmse, rot_rmse = {}, {}
        for camera in self.camera_names:
            translation_errors, rotation_errors = [], []
            for sample in self.samples:
                ref_probe = sample[self.reference_camera]
                predicted = extrinsics[camera] @ sample[camera]
                translation_errors.append(np.linalg.norm(ref_probe[:3, 3] - predicted[:3, 3]))
                rotation_errors.append(np.degrees(np.linalg.norm(
                    rotation_error(ref_probe[:3, :3], predicted[:3, :3]))))
            trans_rmse[camera] = float(np.sqrt(np.mean(np.square(translation_errors))))
            rot_rmse[camera] = float(np.sqrt(np.mean(np.square(rotation_errors))))
        return ExtrinsicCalibrationResult(
            self.reference_camera, extrinsics, len(self.samples), trans_rmse,
            rot_rmse, float(optimum.cost), bool(optimum.success), str(optimum.message))


def save_consensus_yaml(result: ExtrinsicCalibrationResult, path,
                        world_frame=None, consensus=None):
    """Save transforms directly in the schema consumed by pose_consensus.py."""
    frame = world_frame or f'{result.reference_camera}_color_optical_frame'
    data = {
        'world_frame': frame,
        'reference_camera': result.reference_camera,
        'calibration': {
            'sample_count': result.sample_count,
            'optimizer_cost': result.optimizer_cost,
            'optimizer_success': result.optimizer_success,
            'translation_rmse_m': result.translation_rmse_m,
            'rotation_rmse_deg': result.rotation_rmse_deg,
        },
        'consensus': consensus or {
            'translation_tolerance_m': 0.05,
            'rotation_tolerance_deg': 10.0,
            'min_cameras': 2,
            'failures_before_reset': 3,
        },
        'cameras': {},
    }
    for camera, pose in result.reference_T_camera.items():
        q = Rotation.from_matrix(pose[:3, :3]).as_quat()
        data['cameras'][camera] = {'world_T_camera': {
            'translation': {'x': float(pose[0, 3]), 'y': float(pose[1, 3]),
                            'z': float(pose[2, 3])},
            'rotation_xyzw': {'x': float(q[0]), 'y': float(q[1]),
                              'z': float(q[2]), 'w': float(q[3])},
            'matrix': pose.tolist(),
        }}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
