import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from utils.camera_extrinsics import RelativeCameraCalibrator, invert_pose, save_consensus_yaml


def make_pose(t, rpy_deg):
    pose = np.eye(4)
    pose[:3, 3] = t
    pose[:3, :3] = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    return pose


class CameraExtrinsicTests(unittest.TestCase):
    def test_recovers_known_relative_transform(self):
        rng = np.random.default_rng(7)
        reference_T_camera = make_pose([0.42, -0.18, 0.09], [4, -7, 23])
        calibrator = RelativeCameraCalibrator(
            ['reference', 'other'], 'reference', sample_count=30,
            min_probe_translation_m=0, min_probe_rotation_deg=0)
        for _ in range(30):
            reference_T_probe = make_pose(
                rng.uniform([-0.3, -0.3, 0.4], [0.3, 0.3, 1.0]),
                rng.uniform([-40, -40, -90], [40, 40, 90]))
            camera_T_probe = invert_pose(reference_T_camera) @ reference_T_probe
            self.assertTrue(calibrator.add_sample({
                'reference': reference_T_probe,
                'other': camera_T_probe,
            }))
        result = calibrator.solve()
        np.testing.assert_allclose(result.reference_T_camera['other'],
                                   reference_T_camera, atol=1e-7)
        self.assertLess(result.translation_rmse_m['other'], 1e-7)
        self.assertLess(result.rotation_rmse_deg['other'], 1e-7)

    def test_motion_gate_rejects_duplicate(self):
        calibrator = RelativeCameraCalibrator(['a','b'], 'a', sample_count=5)
        sample = {'a': np.eye(4), 'b': np.eye(4)}
        self.assertTrue(calibrator.add_sample(sample))
        self.assertFalse(calibrator.add_sample(sample))

    def test_yaml_is_consensus_compatible(self):
        calibrator = RelativeCameraCalibrator(
            ['a','b'], 'a', sample_count=3,
            min_probe_translation_m=0, min_probe_rotation_deg=0)
        a_T_b = make_pose([1,2,3], [0,0,10])
        for x in [0.0, .1, .2]:
            a_T_probe = make_pose([x,0,1], [x*20,0,0])
            calibrator.add_sample({'a': a_T_probe,
                                   'b': invert_pose(a_T_b) @ a_T_probe})
        result = calibrator.solve()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.yaml'
            save_consensus_yaml(result, path)
            data = yaml.safe_load(path.read_text())
            self.assertEqual(data['reference_camera'], 'a')
            self.assertIn('world_T_camera', data['cameras']['b'])
            self.assertEqual(len(data['cameras']['b']['world_T_camera']['matrix']), 4)


if __name__ == '__main__':
    unittest.main()
