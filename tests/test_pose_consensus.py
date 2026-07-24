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


if __name__ == "__main__":
    unittest.main()
