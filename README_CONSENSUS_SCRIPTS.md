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


## Axis-aligned camera overlays

Set `consensus.force_align_axis` to `x`, `y`, or `z` for a symmetric object.
The standalone multi-camera visualizer then uses a visualization-only pose for
each camera: the selected object axis is identical in the common world frame,
while each raw camera translation is retained so the overlay stays on the
observed object. The raw tracker state is not modified. The fused ROS pose uses
the same canonical axis. Omit the setting or use `null` for the original behavior.


### Automatic 180-degree twist correction

Use the directed forced axis together with automatic twist correction:

```yaml
consensus:
  force_align_axis: x
  auto_correct_forced_axis_twist: true
```

The X direction is first aligned exactly across cameras. The implementation then
checks Y and Z against the raw camera pose. If both are reversed, that identifies
an avoidable 180-degree twist around X. It applies a local 180-degree rotation
about X, which leaves +X unchanged while flipping Y and Z back. This is not soft
or unoriented-axis alignment: the positive forced-axis arrow still overlaps and
points the same way in all views. Set the parameter to `false` to disable this
automatic branch correction.
