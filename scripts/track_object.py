#!/usr/bin/env python3
"""Standalone multi-camera, multi-object batched 6-DoF tracker."""
import argparse, gc, logging, os, sys, time
from collections import deque
from pathlib import Path
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+"); parser.add_argument("--object")
    parser.add_argument("--cameras", nargs="+"); parser.add_argument("--camera")
    parser.add_argument("--width", type=int, default=1280); parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30); parser.add_argument("--est_refine_iter", type=int, default=2)
    parser.add_argument("--track_refine_iter", type=int, default=5); parser.add_argument("--confidence", type=float, default=.5)
    parser.add_argument("--mesh-origin", action="store_true"); parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()
    objects = args.objects or ([args.object] if args.object else None)
    cameras = args.cameras or ([args.camera] if args.camera else ["default"])
    if not objects: parser.error("provide --objects or --object")
    if len(objects)>5 or len(objects)!=len(set(objects)): parser.error("use 1-5 unique objects")
    if len(cameras)!=len(set(cameras)): parser.error("camera names must be unique")

    from utils.batch_tracking import (MultiCameraCapture, build_multi_camera_state,
        collect_tracking_units, register_all_cameras, track_batch)
    from utils.tracking_utils import draw_multi_camera_vis, load_sam3, print_pose, set_logging_format, set_seed
    set_logging_format(); set_seed(0)
    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title="ObjectTracker")
    tracked, _, refiner, glctx = build_multi_camera_state(objects, cameras, args.debug, args.mesh_origin)
    _, sam = load_sam3(args.confidence)
    capture = MultiCameraCapture(cameras, args.width, args.height, args.fps)
    fps_hist = deque(maxlen=30)

    def reset():
        for obj in tracked:
            for state in obj["camera_states"]: state.update(initialized=False, pose=None, pose_last=None)
        fps_hist.clear(); gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    try:
        while True:
            frames = capture.read()
            if frames is None: continue
            if not all(s["initialized"] for o in tracked for s in o["camera_states"]):
                register_all_cameras(tracked, frames, sam, args.est_refine_iter)
            else:
                t0=time.time(); units=collect_tracking_units(tracked, frames)
                try: track_batch(refiner, glctx, units, args.track_refine_iter)
                except Exception as exc:
                    logging.exception("Batch tracking failed: %s", exc)
                    for unit in units: unit.obj["camera_states"][unit.camera_idx].update(initialized=False, pose=None, pose_last=None)
                for unit in units:
                    state=unit.obj["camera_states"][unit.camera_idx]
                    if state["pose"] is not None: print_pose(state["pose"], f"{unit.obj['name']}@{unit.frame['name']}")
                dt=time.time()-t0; fps_hist.append(1/dt if dt>1e-4 else 0)
            if display:
                fps_val=sum(fps_hist)/len(fps_hist) if fps_hist else 0
                key=display.show(draw_multi_camera_vis(frames, tracked, fps_val))
                if key in ("q","Escape"): break
                if key=="r": reset()
    except KeyboardInterrupt: pass
    finally:
        capture.stop()
        if display: display.destroy()

if __name__ == "__main__": main()
