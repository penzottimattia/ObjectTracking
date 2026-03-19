#!/usr/bin/env python3
"""
Standalone 6-DoF multi-object pose tracking (no ROS required).

Uses SAM3 for automatic object detection (text-prompted segmentation)
and FoundationPose for 6-DoF pose estimation and frame-to-frame tracking.
Supports simultaneous tracking of 1–5 objects.

Usage:
    python scripts/track_object.py --objects cup can
    python scripts/track_object.py --objects cup can bottle --confidence 0.4
    python scripts/track_object.py --object cup  # single-object backward compat
"""

import argparse
import logging
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# Reason: append (not insert) so site-packages sam3 is found before
# the project's sam3/ directory which would shadow it as a namespace package.
sys.path.append(str(Path(__file__).resolve().parent.parent))

MAX_OBJECTS = 5

TRACK_COLORS_BGR = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 165, 255),
    (255, 255, 0),
]


def _track_one_object(obj, color_rgb, depth, K, refine_iter):
    """Run FoundationPose track_one for a single object (thread target)."""
    try:
        obj["pose"] = obj["est"].track_one(
            rgb=color_rgb, depth=depth, K=K, iteration=refine_iter,
        )
    except Exception as e:
        logging.warning(f"Tracking lost for '{obj['name']}': {e}")
        obj["initialized"] = False
        obj["pose"] = None


def main():
    """Multi-object tracking: SAM3 detection -> FoundationPose tracking -> console output."""
    parser = argparse.ArgumentParser(
        description="6-DoF multi-object tracking with SAM3 + FoundationPose (standalone)"
    )
    parser.add_argument(
        "--objects", type=str, nargs="+",
        help="Object names to track simultaneously (each must match a folder in object/)",
    )
    parser.add_argument(
        "--object", type=str, default=None,
        help="Single object name (backward compat; prefer --objects)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument(
        "--est_refine_iter", type=int, default=2,
        help="Refinement iterations for initial registration",
    )
    parser.add_argument(
        "--track_refine_iter", type=int, default=2,
        help="Refinement iterations for frame-to-frame tracking",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="SAM3 detection confidence threshold",
    )
    parser.add_argument(
        "--mesh-origin", action="store_true",
        help="Draw axes at the original mesh origin instead of the AABB center",
    )
    parser.add_argument("--debug", type=int, default=1, help="Debug level (0=off, 1=vis, 2=save)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization window")
    parser.add_argument(
        "--mesh-origin", action="store_true",
        help="Draw axes at the original mesh origin instead of the AABB center",
    )
    args = parser.parse_args()

    object_names = args.objects or ([args.object] if args.object else None)
    if not object_names:
        parser.error("Provide --objects <name1> <name2> ... or --object <name>")
    if len(object_names) > MAX_OBJECTS:
        parser.error(f"Maximum {MAX_OBJECTS} objects supported, got {len(object_names)}")
    if len(object_names) != len(set(object_names)):
        parser.error("Duplicate object names are not allowed")

    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title="ObjectTracker")

    # Deferred imports: these trigger CUDA init via FoundationPose/PyTorch
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2 not found. Install: pip install pyrealsense2"
        ) from exc

    from utils.tracking_utils import (
        build_estimator,
        create_shared_estimator_components,
        draw_multi_tracking_vis,
        get_sam3_mask,
        intrinsics_to_K,
        load_mesh,
        load_sam3,
        print_pose,
        set_logging_format,
        set_seed,
    )

    set_logging_format()
    set_seed(0)

    # Reason: glctx (nvdiffrast) is not thread-safe, so each estimator creates
    # its own. Scorer and refiner are shared (read-only in eval mode).
    scorer, refiner, _ = create_shared_estimator_components()

    tracked = []
    for i, name in enumerate(object_names):
        mesh_path, _ = load_mesh(name)
        debug_dir = f"/tmp/fp_debug/{name}"
        os.makedirs(debug_dir, exist_ok=True)
        est, mesh, to_origin, bbox = build_estimator(
            mesh_path=mesh_path,
            debug_dir=debug_dir,
            est_refine_iter=args.est_refine_iter,
            track_refine_iter=args.track_refine_iter,
            debug=args.debug,
            scorer=scorer,
            refiner=refiner,
        )
        tracked.append({
            "name": name,
            "est": est,
            "mesh": mesh,
            "to_origin": to_origin,
            "bbox": bbox,
            "color_bgr": TRACK_COLORS_BGR[i % len(TRACK_COLORS_BGR)],
            "pose": None,
            "initialized": False,
            "mesh_origin": args.mesh_origin,
        })

    _, sam_processor = load_sam3(confidence=args.confidence)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
    except Exception as e:
        logging.error("Failed to start RealSense pipeline: %s", e)
        raise
    align = rs.align(rs.stream.color)

    # Setup signal handlers to allow clean exit on SIGINT/SIGTERM
    import signal

    def _handle_signal(signum, frame):
        logging.info("Received signal %s, shutting down...", signum)
        try:
            if 'pipeline' in locals() and pipeline is not None:
                pipeline.stop()
        except Exception as e:
            logging.warning("Error stopping pipeline during signal handling: %s", e)
        try:
            if display:
                display.destroy()
        except Exception:
            pass
        # exit immediately
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K = intrinsics_to_K(intr)

    names_str = ", ".join(f"'{n}'" for n in object_names)
    logging.info(
        f"RealSense started: {args.width}x{args.height}@{args.fps}fps, "
        f"depth_scale={depth_scale}"
    )
    logging.info(f"Tracking {len(object_names)} object(s): {names_str}")

    fps_hist: deque = deque(maxlen=30)
    track_pool = ThreadPoolExecutor(max_workers=MAX_OBJECTS)

    try:
        running = True
        while running:
            # --- Phase 1: SAM3 detection for uninitialized objects ---
            logging.info("Running SAM3 for initial detection...")

            while not all(o["initialized"] for o in tracked):
                if display and display.closed:
                    running = False
                    break

                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

                for obj in tracked:
                    if obj["initialized"]:
                        continue
                    mask = get_sam3_mask(sam_processor, color_rgb, obj["name"])
                    if mask is not None and mask.sum() > 100:
                        try:
                            obj["pose"] = obj["est"].register(
                                K=K,
                                rgb=color_rgb,
                                depth=depth,
                                ob_mask=mask.astype(bool),
                                iteration=args.est_refine_iter,
                            )
                            obj["initialized"] = True
                            n_done = sum(1 for o in tracked if o["initialized"])
                            logging.info(
                                f"Registered '{obj['name']}' "
                                f"({n_done}/{len(tracked)})"
                            )
                        except Exception as e:
                            logging.warning(
                                f"Registration failed for '{obj['name']}': {e}"
                            )

                if display:
                    vis_bgr = draw_multi_tracking_vis(color_bgr, tracked, K, 0.0)
                    key = display.show(vis_bgr)
                    if key in ("q", "Escape"):
                        running = False
                        break

            if not running:
                break

            # --- Phase 2: real-time FoundationPose tracking (no SAM3) ---
            logging.info("All objects registered — entering real-time tracking loop")

            while True:
                if display and display.closed:
                    running = False
                    break

                frames = pipeline.wait_for_frames()
                frames = align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
                color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
                t0 = time.time()

                active = [o for o in tracked if o["initialized"]]
                if len(active) == 1:
                    _track_one_object(
                        active[0], color_rgb, depth, K, args.track_refine_iter,
                    )
                elif active:
                    futures = [
                        track_pool.submit(
                            _track_one_object, obj, color_rgb, depth, K,
                            args.track_refine_iter,
                        )
                        for obj in active
                    ]
                    for f in futures:
                        f.result()

                for obj in tracked:
                    if obj["initialized"] and obj["pose"] is not None:
                        print_pose(obj["pose"], obj["name"])

                dt = time.time() - t0
                fps_hist.append(1.0 / dt if dt > 1e-4 else 0.0)
                fps_val = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

                if display:
                    vis_bgr = draw_multi_tracking_vis(color_bgr, tracked, K, fps_val)
                    key = display.show(vis_bgr)
                    if key in ("q", "Escape"):
                        running = False
                        break
                    if key == "r":
                        for obj in tracked:
                            obj["initialized"] = False
                            obj["pose"] = None
                        fps_hist.clear()
                        logging.info("Tracking reset — re-detecting all objects")
                        break

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        track_pool.shutdown(wait=False)
        pipeline.stop()
        if display:
            display.destroy()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()