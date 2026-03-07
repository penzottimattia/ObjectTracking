"""
Tests for scripts/tracking_utils.py and scripts/track_object_ros.py.

Covers: mesh loading, intrinsics conversion, pose printing,
SAM3 mask selection, and error handling for missing objects.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "FoundationPose"))


# ---------------------------------------------------------------------------
# load_mesh
# ---------------------------------------------------------------------------


class TestLoadMesh:
    """Tests for load_mesh()."""

    def test_valid_object(self):
        """Expected use: loading a known object returns a valid path."""
        from tracking_utils import load_mesh

        mesh_path, mesh_dir = load_mesh("cup")
        assert mesh_path.endswith(".obj")
        assert Path(mesh_path).exists()
        assert mesh_dir.exists()

    def test_missing_object_exits(self):
        """Failure case: non-existent object triggers sys.exit."""
        from tracking_utils import load_mesh

        with pytest.raises(SystemExit):
            load_mesh("nonexistent_object_xyz_12345")

    def test_all_objects_have_obj_file(self):
        """Edge case: every folder in object/ has at least one .obj file."""
        object_root = PROJECT_ROOT / "object"
        if not object_root.exists():
            pytest.skip("object/ directory not present")

        for obj_dir in object_root.iterdir():
            if obj_dir.is_dir():
                obj_files = list(obj_dir.glob("*.obj"))
                assert len(obj_files) > 0, f"No .obj file in {obj_dir.name}"


# ---------------------------------------------------------------------------
# intrinsics_to_K
# ---------------------------------------------------------------------------


class TestIntrinsicsToK:
    """Tests for intrinsics_to_K()."""

    def test_expected_output(self):
        """Expected use: converts RealSense-like intrinsics to 3x3 matrix."""
        from tracking_utils import intrinsics_to_K

        mock_intr = MagicMock()
        mock_intr.fx = 615.0
        mock_intr.fy = 615.0
        mock_intr.ppx = 320.0
        mock_intr.ppy = 240.0

        K = intrinsics_to_K(mock_intr)
        assert K.shape == (3, 3)
        assert K.dtype == np.float32
        assert K[0, 0] == pytest.approx(615.0)
        assert K[1, 1] == pytest.approx(615.0)
        assert K[0, 2] == pytest.approx(320.0)
        assert K[1, 2] == pytest.approx(240.0)
        assert K[2, 2] == pytest.approx(1.0)

    def test_zero_principal_point(self):
        """Edge case: principal point at origin."""
        from tracking_utils import intrinsics_to_K

        mock_intr = MagicMock()
        mock_intr.fx = 500.0
        mock_intr.fy = 500.0
        mock_intr.ppx = 0.0
        mock_intr.ppy = 0.0

        K = intrinsics_to_K(mock_intr)
        assert K[0, 2] == pytest.approx(0.0)
        assert K[1, 2] == pytest.approx(0.0)

    def test_off_diagonal_zeros(self):
        """Expected: off-diagonal elements (except [0,2], [1,2]) are zero."""
        from tracking_utils import intrinsics_to_K

        mock_intr = MagicMock()
        mock_intr.fx = 600.0
        mock_intr.fy = 600.0
        mock_intr.ppx = 320.0
        mock_intr.ppy = 240.0

        K = intrinsics_to_K(mock_intr)
        assert K[0, 1] == pytest.approx(0.0)
        assert K[1, 0] == pytest.approx(0.0)
        assert K[2, 0] == pytest.approx(0.0)
        assert K[2, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# print_pose
# ---------------------------------------------------------------------------


class TestPrintPose:
    """Tests for print_pose() — console output."""

    def test_identity_pose(self):
        """Expected use: identity matrix prints without errors."""
        from tracking_utils import print_pose

        pose = np.eye(4)
        print_pose(pose, "test_object")

    def test_arbitrary_pose(self):
        """Expected use: arbitrary valid pose prints without errors."""
        from tracking_utils import print_pose

        pose = np.eye(4)
        pose[:3, 3] = [0.1, -0.2, 0.5]
        print_pose(pose, "cup")

    def test_singular_rotation_does_not_crash(self):
        """Edge case: near-singular rotation matrix doesn't crash."""
        from tracking_utils import print_pose

        pose = np.eye(4)
        pose[0, 0] = -1.0
        pose[1, 1] = -1.0
        print_pose(pose, "edge_case")


# ---------------------------------------------------------------------------
# get_sam3_mask
# ---------------------------------------------------------------------------


class TestGetSam3Mask:
    """Tests for get_sam3_mask() with mocked SAM3."""

    def _make_mock_processor(self, n_masks: int, score: float = 0.9):
        """Helper to build a mock Sam3Processor."""
        import torch

        processor = MagicMock()

        masks = torch.zeros((n_masks, 1, 480, 640), dtype=torch.bool)
        if n_masks > 0:
            masks[0, 0, 100:200, 100:200] = True
        scores = torch.full((n_masks,), score)

        def mock_set_image(img):
            return {"state": "ok"}

        def mock_set_text_prompt(state, prompt):
            return {"masks": masks, "scores": scores, "boxes": None}

        processor.set_image = mock_set_image
        processor.set_text_prompt = mock_set_text_prompt
        return processor

    @patch("tracking_utils.Image", create=True)
    def test_returns_mask_on_detection(self, mock_pil):
        """Expected use: returns binary mask when object is detected."""
        from tracking_utils import get_sam3_mask

        mock_pil.fromarray = MagicMock(return_value=MagicMock())
        processor = self._make_mock_processor(n_masks=1, score=0.9)
        color_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("tracking_utils.Image") as pil_mock:
            pil_mock.fromarray.return_value = MagicMock()
            mask = get_sam3_mask(processor, color_rgb, "cup")

        assert mask is not None
        assert mask.dtype == np.uint8
        assert mask.shape == (480, 640)
        assert mask.sum() > 0

    @patch("tracking_utils.Image", create=True)
    def test_returns_none_on_no_detection(self, mock_pil):
        """Failure case: returns None when SAM3 finds nothing."""
        from tracking_utils import get_sam3_mask

        mock_pil.fromarray = MagicMock(return_value=MagicMock())
        processor = self._make_mock_processor(n_masks=0)
        color_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("tracking_utils.Image") as pil_mock:
            pil_mock.fromarray.return_value = MagicMock()
            mask = get_sam3_mask(processor, color_rgb, "nonexistent")

        assert mask is None

    @patch("tracking_utils.Image", create=True)
    def test_picks_highest_score(self, mock_pil):
        """Edge case: with multiple detections, picks the highest score."""
        import torch
        from tracking_utils import get_sam3_mask

        processor = MagicMock()
        masks = torch.zeros((3, 1, 480, 640), dtype=torch.bool)
        masks[0, 0, 10:20, 10:20] = True
        masks[1, 0, 100:300, 100:300] = True
        masks[2, 0, 50:60, 50:60] = True
        scores = torch.tensor([0.5, 0.95, 0.7])

        processor.set_image = MagicMock(return_value={})
        processor.set_text_prompt = MagicMock(
            return_value={"masks": masks, "scores": scores, "boxes": None}
        )
        mock_pil.fromarray = MagicMock(return_value=MagicMock())

        color_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("tracking_utils.Image") as pil_mock:
            pil_mock.fromarray.return_value = MagicMock()
            mask = get_sam3_mask(processor, color_rgb, "cup")

        assert mask is not None
        assert mask[150, 150] == 1
        assert mask[5, 5] == 0


# ---------------------------------------------------------------------------
# pose_to_msg (ROS version)
# ---------------------------------------------------------------------------


class TestPoseToMsg:
    """Tests for pose_to_msg() in track_object_ros.py."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_ros(self):
        """Skip these tests if ROS is not available."""
        try:
            import rospy  # noqa: F401
            from geometry_msgs.msg import PoseStamped  # noqa: F401
        except ImportError:
            pytest.skip("ROS not installed")

    def test_identity_pose_msg(self):
        """Expected use: identity pose produces valid PoseStamped."""
        from track_object_ros import pose_to_msg

        pose = np.eye(4)
        with patch("track_object_ros.rospy") as mock_rospy:
            mock_rospy.Time.now.return_value = MagicMock()
            msg = pose_to_msg(pose, "camera_frame")

        assert msg.pose.position.x == pytest.approx(0.0)
        assert msg.pose.position.y == pytest.approx(0.0)
        assert msg.pose.position.z == pytest.approx(0.0)
        assert msg.header.frame_id == "camera_frame"

    def test_translated_pose_msg(self):
        """Expected use: non-zero translation is reflected in message."""
        from track_object_ros import pose_to_msg

        pose = np.eye(4)
        pose[:3, 3] = [0.5, -0.3, 1.2]
        with patch("track_object_ros.rospy") as mock_rospy:
            mock_rospy.Time.now.return_value = MagicMock()
            msg = pose_to_msg(pose, "cam")

        assert msg.pose.position.x == pytest.approx(0.5)
        assert msg.pose.position.y == pytest.approx(-0.3)
        assert msg.pose.position.z == pytest.approx(1.2)

    def test_quaternion_is_unit(self):
        """Edge case: quaternion in message is always unit length."""
        from track_object_ros import pose_to_msg

        pose = np.eye(4)
        with patch("track_object_ros.rospy") as mock_rospy:
            mock_rospy.Time.now.return_value = MagicMock()
            msg = pose_to_msg(pose, "cam")

        q = msg.pose.orientation
        norm = np.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        assert norm == pytest.approx(1.0, abs=1e-6)
