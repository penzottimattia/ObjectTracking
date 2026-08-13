#!/usr/bin/env python3
"""Standalone multi-camera, multi-object batched 6-DoF tracker."""
import argparse
import gc
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(str(Path(__file__).resolve().parent.parent))


def add_consensus_arguments(parser):
    parser.add_argument(
        "--consensus-config", default=None,
        help="Enable multi-camera consensus using world_T_camera transforms from YAML",
    )
    parser.add_argument(
        "--consensus-translation", type=float, default=None,
        help="Override translation disagreement tolerance in metres",
    )
    parser.add_argument(
        "--consensus-rotation-deg", type=float, default=None,
        help="Override rotation disagreement tolerance in degrees",
    )
    parser.add_argument(
        "--consensus-failures", type=int, default=None,
        help="Override consecutive disagreements required before reset",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batched multi-camera 6-DoF tracking with optional pose consensus"
    )
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--object")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--est_refine_iter", type=int, default=2)
    parser.add_argument("--track_refine_iter", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--mesh-origin", action="store_true")
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--no-vis", action="store_true")
    add_consensus_arguments(parser)
    args = parser.parse_args()

    objects = args.objects or ([args.object] if args.object else None)
    cameras = args.cameras or ([args.camera] if args.camera else ["default"])
    if not objects:
        parser.error("provide --objects or --object")
    if len(objects) > 5 or len(objects) != len(set(objects)):
        parser.error("use 1-5 unique objects")
    if len(cameras) != len(set(cameras)):
        parser.error("camera names must be unique")
    if args.consensus_config and len(cameras) < 2:
        parser.error("consensus requires at least two cameras")

    from utils.batch_tracking import (
        MultiCameraCapture,
        build_multi_camera_state,
        collect_tracking_units,
        register_all_cameras,
        track_batch,
    )
    from utils.tracking_utils import (
        draw_multi_camera_vis,
        load_sam3,
        print_pose,
        set_logging_format,
        set_seed,
    )

    set_logging_format()
    set_seed(0)

    consensus = None
    if args.consensus_config:
        from utils.pose_consensus import PoseConsensus
        consensus = PoseConsensus.from_yaml(
            args.consensus_config,
            translation_tolerance_m=args.consensus_translation,
            rotation_tolerance_deg=args.consensus_rotation_deg,
            failures_before_reset=args.consensus_failures,
        )
        missing = sorted(set(cameras) - set(consensus.world_T_camera))
        if missing:
            parser.error(f"consensus config has no transform for cameras: {missing}")
        logging.info(
            "Consensus enabled in '%s' frame for cameras: %s",
            consensus.world_frame, ", ".join(cameras),
        )

    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title="ObjectTracker")

    tracked, _, refiner, glctx = build_multi_camera_state(
        objects, cameras, args.debug, args.mesh_origin
    )
    _, sam = load_sam3(args.confidence)
    capture = MultiCameraCapture(cameras, args.width, args.height, args.fps)
    fps_hist = deque(maxlen=30)

    def reset(reason="requested"):
        logging.warning("Resetting all trackers: %s", reason)
        for obj in tracked:
            for state in obj["camera_states"]:
                state.update(initialized=False, pose=None, pose_last=None)
        if consensus is not None:
            consensus.reset_state()
        fps_hist.clear()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        while True:
            frames = capture.read()
            if frames is None:
                continue

            all_initialized = all(
                state["initialized"]
                for obj in tracked
                for state in obj["camera_states"]
            )
            if not all_initialized:
                register_all_cameras(
                    tracked, frames, sam, args.est_refine_iter
                )
            else:
                t0 = time.time()
                units = collect_tracking_units(tracked, frames)
                try:
                    track_batch(refiner, glctx, units, args.track_refine_iter)
                except Exception as exc:
                    logging.exception("Batch tracking failed: %s", exc)
                    reset("batch tracking failure")
                    continue

                consensus_results = {}
                if consensus is not None:
                    from utils.consensus_tracking import (
                        evaluate_tracked_objects,
                        should_reset_any,
                        update_consensus_display_poses,
                    )
                    consensus_results = evaluate_tracked_objects(
                        consensus, tracked, cameras
                    )
                    update_consensus_display_poses(
                        consensus, tracked, cameras, consensus_results
                    )
                    if should_reset_any(consensus_results):
                        detail = "; ".join(
                            f"{name}: {result.translation_error_m:.4f} m, "
                            f"{result.rotation_error_deg:.2f} deg"
                            for name, result in consensus_results.items()
                            if result.should_reset
                        )
                        reset(f"pose consensus failure ({detail})")
                        continue

                    for object_name, result in consensus_results.items():
                        if result.consistent and result.pose_world is not None:
                            print_pose(
                                result.pose_world,
                                f"{object_name}@{consensus.world_frame}",
                            )
                else:
                    for unit in units:
                        state = unit.obj["camera_states"][unit.camera_idx]
                        if state["pose"] is not None:
                            print_pose(
                                state["pose"],
                                f"{unit.obj['name']}@{unit.frame['name']}",
                            )

                elapsed = time.time() - t0
                fps_hist.append(1.0 / elapsed if elapsed > 1e-4 else 0.0)

            if display:
                fps_value = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0
                key = display.show(draw_multi_camera_vis(frames, tracked, fps_value))
                if key in ("q", "Escape"):
                    break
                if key == "r":
                    reset("keyboard")
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        capture.stop()
        if display:
            display.destroy()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()
