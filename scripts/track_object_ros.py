#!/usr/bin/env python3
"""ROS2 multi-camera, multi-object batched tracker with optional consensus."""
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
        help="Enable consensus using world_T_camera transforms from YAML",
    )
    parser.add_argument("--consensus-translation", type=float, default=None)
    parser.add_argument("--consensus-rotation-deg", type=float, default=None)
    parser.add_argument("--consensus-failures", type=int, default=None)
    parser.add_argument(
        "--fused-topic-prefix", default="/object_pose_fused",
        help="Prefix for accepted common-frame poses",
    )
    parser.add_argument(
        "--publish-camera-poses", action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish individual camera-frame poses as well as fused poses",
    )


def main():
    parser = argparse.ArgumentParser(
        description="ROS2 batched multi-camera tracker with optional consensus"
    )
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--object")
    parser.add_argument("--cameras", nargs="+")
    parser.add_argument("--camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--ros-rate", "--hz", dest="ros_rate", type=float, default=0.0)
    parser.add_argument("--est_refine_iter", type=int, default=2)
    parser.add_argument("--track_refine_iter", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--frame_id", default=None, help="Single-camera override")
    parser.add_argument("--topic", default=None, help="Legacy single-camera topic")
    parser.add_argument("--topic-prefix", default="/object_pose")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--reset-service", default="/reset_tracker")
    parser.add_argument("--track-frames", type=int, default=0)
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--mesh-origin", action="store_true")
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

    import rclpy
    from cv_bridge import CvBridge
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_srvs.srv import Trigger
    from scipy.spatial.transform import Rotation as R

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

    rclpy.init()
    node = rclpy.create_node("object_tracker")
    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title="ObjectTracker")

    tracked, _, refiner, glctx = build_multi_camera_state(
        objects, cameras, args.debug, args.mesh_origin
    )
    _, sam = load_sam3(args.confidence)
    capture = MultiCameraCapture(cameras, args.width, args.height, args.fps)
    bridge = CvBridge()

    camera_pose_publishers = {}
    if args.publish_camera_poses:
        for camera_index, camera in enumerate(cameras):
            for obj in tracked:
                if len(cameras) == 1 and len(tracked) == 1 and args.topic:
                    topic = args.topic
                else:
                    topic = f"{args.topic_prefix.rstrip('/')}/{camera}/{obj['name']}"
                camera_pose_publishers[(camera_index, obj["name"])] = (
                    node.create_publisher(PoseStamped, topic, 10)
                )
                logging.info("Camera pose topic: %s", topic)

    fused_publishers = {}
    if consensus is not None:
        for obj in tracked:
            topic = f"{args.fused_topic_prefix.rstrip('/')}/{obj['name']}"
            fused_publishers[obj["name"]] = node.create_publisher(
                PoseStamped, topic, 10
            )
            logging.info(
                "Fused pose topic: %s (frame: %s)", topic, consensus.world_frame
            )

    image_publishers = {}
    for camera_index, camera in enumerate(cameras):
        topic = (
            args.image_topic if len(cameras) == 1
            else f"{args.image_topic.rstrip('/')}/{camera}"
        )
        image_publishers[camera_index] = node.create_publisher(Image, topic, 10)

    def pose_msg(pose, frame_id):
        message = PoseStamped()
        message.header.frame_id = frame_id
        message.header.stamp = node.get_clock().now().to_msg()
        message.pose.position.x = float(pose[0, 3])
        message.pose.position.y = float(pose[1, 3])
        message.pose.position.z = float(pose[2, 3])
        quaternion = R.from_matrix(pose[:3, :3]).as_quat()
        message.pose.orientation.x = float(quaternion[0])
        message.pose.orientation.y = float(quaternion[1])
        message.pose.orientation.z = float(quaternion[2])
        message.pose.orientation.w = float(quaternion[3])
        return message

    pending_reset = {"value": False, "reason": "service"}
    paused = False
    frames_remaining = args.track_frames if args.track_frames > 0 else None
    fps_hist = deque(maxlen=30)
    next_publish_time = 0.0

    def reset(reason="requested"):
        nonlocal paused, frames_remaining
        logging.warning("Resetting all trackers: %s", reason)
        for obj in tracked:
            for state in obj["camera_states"]:
                state.update(initialized=False, pose=None, pose_last=None)
        if consensus is not None:
            consensus.reset_state()
        paused = False
        frames_remaining = args.track_frames if args.track_frames > 0 else None
        fps_hist.clear()
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def reset_service(request, response):
        del request
        pending_reset["value"] = True
        pending_reset["reason"] = "external reset service"
        response.success = True
        response.message = "Reset queued"
        return response

    node.create_service(Trigger, args.reset_service, reset_service)
    logging.info("Reset service: %s", args.reset_service)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            if pending_reset["value"]:
                pending_reset["value"] = False
                reset(pending_reset["reason"])

            if paused:
                if display:
                    display.pump()
                    if display.closed:
                        break
                time.sleep(0.05)
                continue

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
                started = time.time()
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
                    )
                    consensus_results = evaluate_tracked_objects(
                        consensus, tracked, cameras
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

                now = time.time()
                publish_now = (
                    args.ros_rate <= 0.0 or now >= next_publish_time
                )
                if publish_now:
                    next_publish_time = (
                        now + 1.0 / args.ros_rate if args.ros_rate > 0.0 else now
                    )

                    if args.publish_camera_poses:
                        for unit in units:
                            state = unit.obj["camera_states"][unit.camera_idx]
                            if state["pose"] is None:
                                continue
                            frame_id = (
                                args.frame_id
                                if len(cameras) == 1 and args.frame_id
                                else unit.frame["frame_id"]
                            )
                            publisher = camera_pose_publishers[
                                (unit.camera_idx, unit.obj["name"])
                            ]
                            publisher.publish(pose_msg(state["pose"], frame_id))

                    for object_name, result in consensus_results.items():
                        if result.consistent and result.pose_world is not None:
                            fused_publishers[object_name].publish(
                                pose_msg(result.pose_world, consensus.world_frame)
                            )

                    for camera_index, frame in enumerate(frames):
                        image_message = bridge.cv2_to_imgmsg(
                            frame["color_bgr"], encoding="bgr8"
                        )
                        image_message.header.frame_id = frame["frame_id"]
                        image_message.header.stamp = node.get_clock().now().to_msg()
                        image_publishers[camera_index].publish(image_message)

                elapsed = time.time() - started
                fps_hist.append(1.0 / elapsed if elapsed > 1e-4 else 0.0)

                if frames_remaining is not None:
                    frames_remaining -= 1
                    if frames_remaining <= 0:
                        paused = True
                        logging.info("Track-frame limit reached; waiting for reset")

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
        node.destroy_node()
        rclpy.shutdown()
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()
