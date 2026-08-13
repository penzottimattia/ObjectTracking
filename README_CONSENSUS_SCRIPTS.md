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

## Symmetric objects: forced axis alignment

Quaternion averaging can publish an in-between orientation when cameras select
different equivalent symmetry branches, for example on a hexagonal object. Set
`force_align_axis` in the YAML `consensus` section to `x`, `y`, or `z`:

```yaml
consensus:
  force_align_axis: x
```

The selected object-frame axis is copied exactly from the first contributing
camera in deterministic camera-name order. Translation and the remaining
rotational degree of freedom are still fused. Leave it `null` or omit it to keep
the previous quaternion-average behavior. Choose an axis that distinguishes the
symmetry branch (for a prism symmetric around Z, usually choose X or Y).

On confirmed disagreement, the scripts call the same internal reset routine used by keyboard and ROS service resets. All camera/object states are cleared, GPU cache cleanup is requested, and SAM3 registration starts again.
