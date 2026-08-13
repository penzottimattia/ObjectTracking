import unittest
import numpy as np
from utils.pose_consensus import PoseConsensus, average_poses


def pose(x=0, yaw_deg=0):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    value = np.eye(4)
    value[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    value[0, 3] = x
    return value


def _rotation_distance_for_test(a, b):
    relative = a.T @ b
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PoseConsensusTests(unittest.TestCase):
    def test_consistent_poses_are_averaged(self):
        engine = PoseConsensus({"a": np.eye(4), "b": np.eye(4)},
                               translation_tolerance_m=.05,
                               rotation_tolerance_deg=5)
        result = engine.evaluate("cup", {"a": pose(1.00), "b": pose(1.02)})
        self.assertTrue(result.consistent)
        self.assertFalse(result.should_reset)
        self.assertAlmostEqual(result.pose_world[0, 3], 1.01)

    def test_world_camera_transform_is_applied(self):
        world_T_b = pose(1.0)
        engine = PoseConsensus({"a": np.eye(4), "b": world_T_b})
        result = engine.evaluate("cup", {"a": pose(2.0), "b": pose(1.0)})
        self.assertTrue(result.consistent)
        self.assertAlmostEqual(result.pose_world[0, 3], 2.0)

    def test_repeated_disagreement_requests_reset(self):
        engine = PoseConsensus({"a": np.eye(4), "b": np.eye(4)},
                               translation_tolerance_m=.05,
                               failures_before_reset=2)
        first = engine.evaluate("cup", {"a": pose(0), "b": pose(1)})
        second = engine.evaluate("cup", {"a": pose(0), "b": pose(1)})
        self.assertFalse(first.should_reset)
        self.assertTrue(second.should_reset)
        self.assertIsNone(second.pose_world)

    def test_rotation_disagreement(self):
        engine = PoseConsensus({"a": np.eye(4), "b": np.eye(4)},
                               rotation_tolerance_deg=5)
        result = engine.evaluate("cup", {"a": pose(yaw_deg=0), "b": pose(yaw_deg=20)})
        self.assertFalse(result.consistent)
        self.assertAlmostEqual(result.rotation_error_deg, 20.0, places=5)

    def test_quaternion_sign_does_not_break_average(self):
        result = average_poses([pose(yaw_deg=179), pose(yaw_deg=-179)])
        self.assertAlmostEqual(result[0, 0], -1.0, places=3)

    def test_display_pose_aligns_selected_axis_across_camera_frames(self):
        world_T_b = pose(0.0, yaw_deg=90)
        engine = PoseConsensus(
            {"a": np.eye(4), "b": world_T_b},
            rotation_tolerance_deg=90, force_align_axis="x"
        )
        raw_a = pose(1.0, yaw_deg=0)
        raw_b = pose(1.0, yaw_deg=-30)
        fused_world = pose(1.0, yaw_deg=0)
        display_a = engine.align_camera_pose_for_display("a", raw_a, fused_world)
        display_b = engine.align_camera_pose_for_display("b", raw_b, fused_world)
        axis_a_world = (engine.world_T_camera["a"] @ display_a)[:3, 0]
        axis_b_world = (engine.world_T_camera["b"] @ display_b)[:3, 0]
        np.testing.assert_allclose(axis_a_world, axis_b_world, atol=1e-9)
        np.testing.assert_allclose(display_b[:3, 3], raw_b[:3, 3], atol=1e-9)

    def test_invalid_force_align_axis_is_rejected(self):
        with self.assertRaises(ValueError):
            PoseConsensus({"a": np.eye(4), "b": np.eye(4)}, force_align_axis="yaw")

    def test_auto_twist_correction_flips_non_forced_axes_only(self):
        engine = PoseConsensus(
            {"a": np.eye(4), "b": np.eye(4)},
            force_align_axis="x", auto_correct_forced_axis_twist=True
        )
        raw = np.eye(4)
        # Target has the same +X but Y and Z are both reversed: a pi twist on X.
        target = np.eye(4)
        target[:3, :3] = np.diag([1.0, -1.0, -1.0])
        display = engine.align_camera_pose_for_display("b", raw, target)
        np.testing.assert_allclose(display[:3, 0], target[:3, 0], atol=1e-9)
        np.testing.assert_allclose(display[:3, 1], raw[:3, 1], atol=1e-9)
        np.testing.assert_allclose(display[:3, 2], raw[:3, 2], atol=1e-9)

    def test_auto_twist_correction_can_be_disabled(self):
        engine = PoseConsensus(
            {"a": np.eye(4), "b": np.eye(4)},
            force_align_axis="x", auto_correct_forced_axis_twist=False
        )
        raw = np.eye(4)
        target = np.eye(4)
        target[:3, :3] = np.diag([1.0, -1.0, -1.0])
        display = engine.align_camera_pose_for_display("b", raw, target)
        np.testing.assert_allclose(display[:3, 0], target[:3, 0], atol=1e-9)
        np.testing.assert_allclose(display[:3, 1], target[:3, 1], atol=1e-9)
        np.testing.assert_allclose(display[:3, 2], target[:3, 2], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
