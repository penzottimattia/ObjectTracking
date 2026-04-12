#!/usr/bin/env python3
"""
ROS2 node for real-time 6-DoF object pose tracking.

Uses SAM3 for automatic object detection (text-prompted segmentation)
and FoundationPose for 6-DoF pose estimation and frame-to-frame tracking.
Publishes pose as geometry_msgs/PoseStamped in the camera optical frame.

Usage:
    python scripts/track_object_ros.py --object cup
    python scripts/track_object_ros.py --object can --topic /can_pose
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


def main():
    """Main tracking loop: SAM3 detection -> FoundationPose tracking -> ROS2 publish."""
    parser = argparse.ArgumentParser(
        description="6-DoF object tracking with SAM3 + FoundationPose, published via ROS2"
    )
    parser.add_argument(
        "--objects", type=str, nargs="+",
        help="Object names (must match folders in object/)",
    )
    parser.add_argument(
        "--object", type=str, default=None,
        help="Single object name (backward compat; prefer --objects)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Camera width")
    parser.add_argument("--height", type=int, default=720, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument(
        "--est_refine_iter", type=int, default=2,
        help="Refinement iterations for initial registration",
    )
    parser.add_argument(
        "--track_refine_iter", type=int, default=5,
        help="Refinement iterations for frame-to-frame tracking (more = less drift, slower)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="SAM3 detection confidence threshold",
    )
    parser.add_argument(
        "--frame_id", type=str, default="camera_color_optical_frame",
        help="ROS2 TF frame ID for the published pose",
    )
    parser.add_argument(
        "--topic", type=str, default="/object_pose",
        help="ROS2 topic to publish PoseStamped on",
    )
    parser.add_argument("--debug", type=int, default=1, help="Debug level (0=off, 1=vis, 2=save)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization window")
    parser.add_argument(
        "--camera", type=str, default=None,
        help="Camera name from camera_config.yaml (e.g. robotcam, tablecam). "
             "Default uses the 'default' key in the config.",
    )
    parser.add_argument(
        "--mesh-origin", action="store_true",
        help="Draw axes at the original mesh origin instead of the AABB center",
    )
    parser.add_argument(
        "--image-topic", type=str, default="/camera/color/image_raw",
        help="ROS2 topic to publish camera images on",
    )
    parser.add_argument(
        "--reset-service", type=str, default="/reset_tracker",
        help="ROS2 Trigger service name for external tracking reset",
    )
    parser.add_argument(
        "--track-frames", type=int, default=0,
        help="Number of tracking frames to run before pausing until reset is called. 0 disables the limit.",
    )
    args = parser.parse_args()

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

    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_srvs.srv import Trigger
    from cv_bridge import CvBridge
    from scipy.spatial.transform import Rotation as R

    from utils.tracking_utils import (
        build_estimator,
        create_shared_estimator_components,
        draw_tracking_vis,
        draw_multi_tracking_vis,
        get_sam3_mask,
        intrinsics_to_K,
        load_camera_serial,
        load_mesh,
        load_sam3,
        set_logging_format,
        set_seed,
    )

    def pose_to_msg(pose: np.ndarray, frame_id: str, node: Node) -> PoseStamped:
        """
        Convert a 4x4 pose matrix to a ROS2 PoseStamped message.

        Args:
            pose (np.ndarray): 4x4 homogeneous transform (object in camera frame).
            frame_id (str): TF frame ID for the header.
            node (Node): ROS2 node (used for clock).

        Returns:
            PoseStamped: ROS2 message with position and orientation.
        """
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = node.get_clock().now().to_msg()

        msg.pose.position.x = float(pose[0, 3])
        msg.pose.position.y = float(pose[1, 3])
        msg.pose.position.z = float(pose[2, 3])

        quat = R.from_matrix(pose[:3, :3]).as_quat()  # [x, y, z, w]
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        return msg

    MAX_OBJECTS = 5

    TRACK_COLORS_BGR = [
        (0, 255, 0),
        (0, 0, 255),
        (255, 0, 0),
        (0, 165, 255),
        (255, 255, 0),
    ]
    def _track_one_object(obj, color_rgb, depth, K, refine_iter):
        try:
            obj["pose"] = obj["est"].track_one(
                rgb=color_rgb, depth=depth, K=K, iteration=refine_iter,
            )
        except Exception as e:
            logging.warning(f"Tracking lost for '{obj['name']}': {e}")
            obj["initialized"] = False
            obj["pose"] = None

    set_logging_format()
    set_seed(0)

    object_names = args.objects or ([args.object] if args.object else None)
    if not object_names:
        parser.error("Provide --objects <name1> <name2> ... or --object <name>")
    if len(object_names) > MAX_OBJECTS:
        parser.error(f"Maximum {MAX_OBJECTS} objects supported, got {len(object_names)}")
    if len(object_names) != len(set(object_names)):
        parser.error("Duplicate object names are not allowed")

    # create shared read-only components and build per-object estimators
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

    # --- ROS2 init ---
    rclpy.init()
    node = rclpy.create_node("object_tracker")

    # Create a publisher per object (single-object backward compat uses args.topic)
    publishers = {}
    if len(tracked) == 1:
        publishers[tracked[0]["name"]] = node.create_publisher(PoseStamped, args.topic, 10)
        logging.info(f"ROS2 publisher on '{args.topic}' (frame: {args.frame_id})")
    else:
        for obj in tracked:
            topic_name = f"{args.topic.rstrip('/')}/{obj['name']}"
            publishers[obj["name"]] = node.create_publisher(PoseStamped, topic_name, 10)
            logging.info(f"ROS2 publisher on '{topic_name}' (frame: {args.frame_id})")

    # Create image publisher
    image_publisher = node.create_publisher(Image, args.image_topic, 10)
    bridge = CvBridge()
    logging.info(f"ROS2 image publisher on '{args.image_topic}'")

    # --- RealSense ---
    pipeline = rs.pipeline()
    config = rs.config()
    serial = load_camera_serial(args.camera)
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    try:
        profile = pipeline.start(config)
    except Exception as e:
        logging.error("Failed to start RealSense pipeline: %s", e)
        # Try to clean up ROS resources before exiting
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        raise
    align = rs.align(rs.stream.color)

    # Setup signal handlers for clean shutdown
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
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
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
    logging.info(
        f"RealSense started: {args.width}x{args.height}@{args.fps}fps, "
        f"depth_scale={depth_scale}"
    )

    fps_hist: deque = deque(maxlen=30)
    track_pool = ThreadPoolExecutor(max_workers=MAX_OBJECTS)
    track_frames_remaining = args.track_frames if args.track_frames > 0 else None
    paused_until_reset = False

    def reset_tracking_state(reason: str) -> None:
        nonlocal track_frames_remaining, paused_until_reset
        for obj in tracked:
            obj["initialized"] = False
            obj["pose"] = None
        fps_hist.clear()
        paused_until_reset = False
        if args.track_frames > 0:
            track_frames_remaining = args.track_frames
        logging.info("Tracking reset (%s) — re-detecting all objects", reason)

    pending_external_reset = {"value": False}

    def handle_reset_service(request, response):
        del request
        pending_external_reset["value"] = True
        response.success = True
        response.message = "Reset requested; tracker will re-detect objects"
        return response

    node.create_service(Trigger, args.reset_service, handle_reset_service)
    logging.info("ROS2 reset service on '%s'", args.reset_service)

    # --- Phase 1: SAM3 detection for uninitialized objects ---
    logging.info("Running SAM3 for initial detection...")

    try:
        running = True
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if pending_external_reset["value"]:
                pending_external_reset["value"] = False
                reset_tracking_state("service")

            if paused_until_reset:
                if display:
                    display.pump()
                    if display.closed:
                        running = False
                        break
                time.sleep(0.05)
                continue

            if display and display.closed:
                break

            # Loop until all objects are initialized
            while not all(o["initialized"] for o in tracked) and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.0)
                if pending_external_reset["value"]:
                    pending_external_reset["value"] = False
                    reset_tracking_state("service")

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
                        valid_depth = (depth >= 0.001) & mask.astype(bool)
                        if valid_depth.sum() < 4:
                            logging.warning(
                                "Invalid RealSense depth for '%s': only %d valid pixels under the SAM3 mask. "
                                "Check the depth stream, camera alignment, and object distance.",
                                obj["name"],
                                int(valid_depth.sum()),
                            )
                            continue
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
                            logging.info(f"Registered '{obj['name']}' ({n_done}/{len(tracked)})")
                        except Exception as e:
                            logging.warning(f"Registration failed for '{obj['name']}': {e}")

                if display:
                    # show all objects (some may be uninitialized)
                    vis_bgr = draw_multi_tracking_vis(color_bgr, tracked, K, 0.0)
                    key = display.show(vis_bgr)
                    if key in ("q", "Escape"):
                        running = False
                        break

            if not running:
                break

            # --- Phase 2: real-time FoundationPose tracking (no SAM3) ---
            logging.info("All objects registered — entering real-time tracking loop")

            while rclpy.ok() and running:
                rclpy.spin_once(node, timeout_sec=0.0)
                if pending_external_reset["value"]:
                    pending_external_reset["value"] = False
                    reset_tracking_state("service")
                    break

                if paused_until_reset:
                    if display:
                        display.pump()
                        if display.closed:
                            running = False
                            break
                    time.sleep(0.05)
                    break

                if track_frames_remaining is not None and track_frames_remaining <= 0:
                    paused_until_reset = True
                    logging.info(
                        "Reached track-frame limit (%d). Pausing until reset is requested.",
                        args.track_frames,
                    )
                    continue

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
                    _track_one_object(active[0], color_rgb, depth, K, args.track_refine_iter)
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

                # Publish poses for initialized objects
                for obj in tracked:
                    if obj["initialized"] and obj["pose"] is not None:
                        msg = pose_to_msg(obj["pose"], args.frame_id, node)
                        publishers[obj["name"]].publish(msg)

                # Publish camera image
                img_msg = bridge.cv2_to_imgmsg(color_bgr, encoding="bgr8")
                img_msg.header.frame_id = args.frame_id
                img_msg.header.stamp = node.get_clock().now().to_msg()
                image_publisher.publish(img_msg)

                dt = time.time() - t0
                fps_hist.append(1.0 / dt if dt > 1e-4 else 0.0)
                fps_val = sum(fps_hist) / len(fps_hist) if fps_hist else 0.0

                if track_frames_remaining is not None:
                    track_frames_remaining -= 1
                    if track_frames_remaining <= 0:
                        paused_until_reset = True
                        logging.info(
                            "Reached track-frame limit (%d). Pausing until reset is requested.",
                            args.track_frames,
                        )
                        break

                if display:
                    vis_bgr = draw_multi_tracking_vis(color_bgr, tracked, K, fps_val)
                    key = display.show(vis_bgr)
                    if key in ("q", "Escape"):
                        running = False
                        break
                    if key == "r":
                        reset_tracking_state("keyboard")

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        try:
            if 'pipeline' in locals() and pipeline is not None:
                pipeline.stop()
        except Exception as e:
            logging.warning("Error stopping pipeline: %s", e)
        try:
            if display:
                display.destroy()
        except Exception as e:
            logging.warning("Error destroying display: %s", e)
        try:
            node.destroy_node()
        except Exception as e:
            logging.warning("Error destroying ROS node: %s", e)
        try:
            rclpy.shutdown()
        except Exception as e:
            logging.warning("Error shutting down rclpy: %s", e)
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()
