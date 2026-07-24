#!/usr/bin/env python3
"""Estimate relative camera extrinsics by manually moving a known probe."""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.append(str(Path(__file__).resolve().parent.parent))


def reset_tracking(tracked):
    for obj in tracked:
        for state in obj['camera_states']:
            state.update(initialized=False, pose=None, pose_last=None)


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate camera poses relative to a reference camera using a tracked probe')
    parser.add_argument('--probe', required=True, help='Known object folder/name used as probe')
    parser.add_argument('--cameras', nargs='+', required=True)
    parser.add_argument('--reference-camera', required=True)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--output', default='config/pose_consensus.yaml')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--est-refine-iter', type=int, default=2)
    parser.add_argument('--track-refine-iter', type=int, default=5)
    parser.add_argument('--sample-period', type=float, default=0.10,
                        help='Minimum seconds between accepted sample attempts')
    parser.add_argument('--min-motion-m', type=float, default=0.01)
    parser.add_argument('--min-motion-deg', type=float, default=3.0)
    parser.add_argument('--translation-scale-m', type=float, default=0.02)
    parser.add_argument('--rotation-scale-deg', type=float, default=3.0)
    parser.add_argument('--robust-loss', choices=['linear','soft_l1','huber','cauchy','arctan'],
                        default='soft_l1')
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--no-vis', action='store_true')
    args = parser.parse_args()
    if args.reference_camera not in args.cameras:
        parser.error('--reference-camera must be included in --cameras')
    if len(args.cameras) < 2:
        parser.error('At least two cameras are required')

    from utils.batch_tracking import (MultiCameraCapture, build_multi_camera_state,
        collect_tracking_units, register_all_cameras, track_batch)
    from utils.camera_extrinsics import RelativeCameraCalibrator, save_consensus_yaml
    from utils.tracking_utils import (draw_multi_camera_vis, load_sam3,
                                      set_logging_format, set_seed)
    set_logging_format(); set_seed(0)
    display = None
    if not args.no_vis:
        from utils.display import TkDisplay
        display = TkDisplay(title='Relative camera calibration')

    tracked, _, refiner, glctx = build_multi_camera_state(
        [args.probe], args.cameras, args.debug, False)
    _, sam = load_sam3(args.confidence)
    capture = MultiCameraCapture(args.cameras, args.width, args.height, args.fps)
    calibrator = RelativeCameraCalibrator(
        args.cameras, args.reference_camera, args.samples,
        args.min_motion_m, args.min_motion_deg,
        args.translation_scale_m, args.rotation_scale_deg, args.robust_loss)
    last_attempt = 0.0
    logging.info('Move the probe through diverse positions and orientations visible to every camera.')

    try:
        while not calibrator.complete:
            frames = capture.read()
            if frames is None:
                continue
            states = tracked[0]['camera_states']
            if not all(state['initialized'] for state in states):
                register_all_cameras(tracked, frames, sam, args.est_refine_iter)
            else:
                try:
                    track_batch(refiner, glctx, collect_tracking_units(tracked, frames),
                                args.track_refine_iter)
                except Exception as exc:
                    logging.warning('Tracking failed, re-registering probe: %s', exc)
                    reset_tracking(tracked)
                    continue
                now = time.monotonic()
                if now - last_attempt >= args.sample_period:
                    last_attempt = now
                    camera_poses = {
                        camera: states[index]['pose']
                        for index, camera in enumerate(args.cameras)
                    }
                    if calibrator.add_sample(camera_poses):
                        logging.info('Accepted calibration sample %d/%d',
                                     len(calibrator.samples), args.samples)
            if display:
                key = display.show(draw_multi_camera_vis(frames, tracked, 0.0))
                if key in ('q', 'Escape'):
                    raise KeyboardInterrupt
                if key == 'r':
                    reset_tracking(tracked)
        result = calibrator.solve()
        save_consensus_yaml(result, args.output)
        logging.info('Calibration written to %s', args.output)
        for camera in args.cameras:
            pose = result.reference_T_camera[camera]
            logging.info('%s: xyz=(%.6f, %.6f, %.6f), translation RMSE=%.4f m, rotation RMSE=%.3f deg',
                         camera, pose[0,3], pose[1,3], pose[2,3],
                         result.translation_rmse_m[camera], result.rotation_rmse_deg[camera])
    except KeyboardInterrupt:
        logging.info('Calibration cancelled after %d accepted samples', len(calibrator.samples))
    finally:
        capture.stop()
        if display:
            display.destroy()


if __name__ == '__main__':
    main()
