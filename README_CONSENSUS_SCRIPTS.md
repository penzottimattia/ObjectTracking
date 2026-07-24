# Tracking scripts with optional pose consensus

These scripts require the earlier multi-camera and consensus modules:

```text
utils/batch_tracking.py
utils/pose_consensus.py
utils/consensus_tracking.py
```

Consensus remains disabled unless `--consensus-config` is supplied.

## Standalone

```bash
python scripts/track_object.py \
  --object cup \
  --cameras robotcam tablecam \
  --consensus-config config/pose_consensus.yaml
```

When consensus is enabled, the standalone script prints only the accepted fused pose in the configured common frame. Without consensus, it prints each camera pose as before.

## ROS2

```bash
python scripts/track_object_ros.py \
  --object cup \
  --cameras robotcam tablecam \
  --consensus-config config/pose_consensus.yaml
```

Default topics:

```text
/object_pose/robotcam/cup
/object_pose/tablecam/cup
/object_pose_fused/cup
```

The fused topic uses the `world_frame` from the consensus YAML. Individual camera topics can be disabled with `--no-publish-camera-poses`.

## Overrides

```bash
--consensus-translation 0.03
--consensus-rotation-deg 7.5
--consensus-failures 5
--fused-topic-prefix /tracked_pose
```

On confirmed disagreement, the scripts call the same internal reset routine used by keyboard and ROS service resets. All camera/object states are cleared, GPU cache cleanup is requested, and SAM3 registration starts again.
