#!/usr/bin/env python3
"""
Standalone 6-DoF object pose tracking (no ROS required).

Uses SAM3 for automatic object detection (text-prompted segmentation)
and FoundationPose for 6-DoF pose estimation and frame-to-frame tracking.
Prints pose (position + quaternion) to the console each frame.

Usage:
    python scripts/track_object.py --object cup
    python scripts/track_object.py --object can --confidence 0.4
"""

import argparse
import logging
import os
import time
from collections import deque

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise ImportError(
        "pyrealsense2 not found. Install: pip install pyrealsense2"
    ) from exc

from tracking_utils import (
    build_estimator,
    check_drift_and_reregister,
    draw_tracking_vis,
    get_sam3_mask,
    intrinsics_to_K,
    load_mesh,
    load_sam3,
    print_pose,
    set_logging_format,
    set_seed,
)


def main():
    """Main tracking loop: SAM3 detection -> FoundationPose tracking -> console output."""
    parser = argparse.ArgumentParser(
        description="6-DoF object tracking with SAM3 + FoundationPose (standalone)"
    )
    parser.add_argument(
        "--object", type=str, required=True,
        help="Object name (must match folder in object/)",
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
        "--redetect_interval", type=float, default=5.0,
        help="Seconds between SAM3 re-detection checks (0 = disabled)",
    )
    parser.add_argument("--debug", type=int, default=1, help="Debug level (0=off, 1=vis, 2=save)")
    parser.add_argument("--no-flip", action="store_true", help="Disable image flip")
    parser.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization")
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    mesh_path, _ = load_mesh(args.object)

    debug_dir = f"/tmp/fp_debug/{args.object}"
    os.makedirs(debug_dir, exist_ok=True)
    est, mesh, to_origin, bbox = build_estimator(
        mesh_path=mesh_path,
        debug_dir=debug_dir,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
    )

    _, sam_processor = load_sam3(confidence=args.confidence)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = (
        profile.get_stream(rs.stream.color)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    K = intrinsics_to_K(intr)
    logging.info(
        f"RealSense started: {args.width}x{args.height}@{args.fps}fps, "
        f"depth_scale={depth_scale}"
    )

    pose = None
    initialized = False
    fps_hist: deque = deque(maxlen=30)
    last_redetect_time = 0.0

    logging.info(f"Tracking '{args.object}' — waiting for SAM3 detection...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

            if not args.no_flip:
                color_bgr = cv2.flip(color_bgr, -1)
                depth = cv2.flip(depth, -1)

            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.time()

            if not initialized:
                mask = get_sam3_mask(sam_processor, color_rgb, args.object)
                if mask is not None and mask.sum() > 100:
                    try:
                        pose = est.register(
                            K=K,
                            rgb=color_rgb,
                            depth=depth,
                            ob_mask=mask.astype(bool),
                            iteration=args.est_refine_iter,
                        )
                        initialized = True
                        last_redetect_time = time.time()
                        logging.info("Initial pose registered successfully")
                    except Exception as e:
                        logging.warning(f"Registration failed: {e}")
            else:
                try:
                    pose = est.track_one(
                        rgb=color_rgb,
                        depth=depth,
                        K=K,
                        iteration=args.track_refine_iter,
                    )
                except Exception as e:
                    logging.warning(f"Tracking failed: {e}")
                    initialized = False
                    continue

                if (
                    args.redetect_interval > 0
                    and (time.time() - last_redetect_time) >= args.redetect_interval
                ):
                    last_redetect_time = time.time()
                    pose = check_drift_and_reregister(
                        est, sam_processor, pose, color_rgb, depth, K,
                        args.object, args.est_refine_iter,
                    )

                print_pose(pose, args.object)

            dt = time.time() - t0
            fps_hist.append(1.0 / dt if dt > 1e-4 else 0.0)
            fps_val = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

            if not args.no_vis:
                vis_bgr = draw_tracking_vis(
                    color_bgr, pose, to_origin, bbox, K,
                    initialized, fps_val, args.object,
                )
                cv2.imshow("ObjectTracker", vis_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    initialized = False
                    pose = None
                    logging.info("Tracking reset — waiting for SAM3 re-detection")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()
