"""Integration helpers between batch_tracking state and pose_consensus."""
import logging
from typing import Dict, Iterable, List

from utils.pose_consensus import ConsensusResult, PoseConsensus


def evaluate_tracked_objects(consensus: PoseConsensus, tracked: Iterable[dict],
                             camera_names: List[str]) -> Dict[str, ConsensusResult]:
    """Evaluate every object from its per-camera tracking states.

    This function has no ROS dependency and is shared by track_object.py and
    track_object_ros.py. It does not reset state itself; the caller remains in
    control of tracker lifecycle and ROS service behavior.
    """
    results = {}
    for obj in tracked:
        camera_poses = {
            camera_name: obj["camera_states"][camera_idx].get("pose")
            for camera_idx, camera_name in enumerate(camera_names)
            if camera_idx < len(obj["camera_states"])
            and obj["camera_states"][camera_idx].get("initialized", False)
        }
        result = consensus.evaluate(obj["name"], camera_poses)
        results[obj["name"]] = result
        if result.available:
            logging.info(
                "Consensus %s: consistent=%s, translation=%.4f m, rotation=%.2f deg, cameras=%s",
                obj["name"], result.consistent, result.translation_error_m,
                result.rotation_error_deg, ",".join(result.cameras_used))
    return results


def should_reset_any(results: Dict[str, ConsensusResult]) -> bool:
    """Return True when any tracked object reaches its reset threshold."""
    return any(result.should_reset for result in results.values())


def update_consensus_display_poses(consensus: PoseConsensus, tracked: Iterable[dict],
                                   camera_names: List[str],
                                   results: Dict[str, ConsensusResult]) -> None:
    """Store visualization-only poses with the configured axis canonicalized."""
    for obj in tracked:
        result = results.get(obj["name"])
        for camera_idx, camera_name in enumerate(camera_names):
            if camera_idx >= len(obj["camera_states"]):
                continue
            state = obj["camera_states"][camera_idx]
            state.pop("display_pose", None)
            raw_pose = state.get("pose")
            if (raw_pose is None or result is None or not result.consistent or
                    result.pose_world is None or consensus.force_align_axis is None):
                continue
            state["display_pose"] = consensus.align_camera_pose_for_display(
                camera_name, raw_pose, result.pose_world
            )
