# Relative camera extrinsic calibration

This optional module estimates every camera pose relative to one selected reference camera by tracking the same known probe while it is moved manually.

## Mathematical model

For sample `k`, the tracker measures `camera_i_T_probe[k]`. For the reference camera `r` and another camera `i`, the unknown constant transform satisfies:

```text
reference_T_probe[k] ~= reference_T_camera_i @ camera_i_T_probe[k]
```

Each individual sample gives an initial estimate:

```text
reference_T_camera_i[k] = reference_T_probe[k] @ inverse(camera_i_T_probe[k])
```

The final result is not a simple element-wise matrix average. The solver performs robust nonlinear least squares on SE(3), jointly minimizing translation and rotation residuals across all accepted samples. The reference transform is fixed to identity to remove gauge freedom.

## Files

```text
utils/camera_extrinsics.py
scripts/calibrate_camera_extrinsics.py
config/camera_extrinsics.yaml.example
tests/test_camera_extrinsics.py
```

The script relies on the multi-camera `utils.batch_tracking` module from the earlier update.

## Run

```bash
python scripts/calibrate_camera_extrinsics.py \
  --probe probe \
  --cameras robotcam tablecam \
  --reference-camera robotcam \
  --samples 100 \
  --output config/pose_consensus.yaml
```

Move the probe slowly through the common field of view. Use diverse translations and rotations. A sample is accepted only when the probe has moved by at least `--min-motion-m` or rotated by at least `--min-motion-deg` relative to the last accepted sample. This avoids collecting many nearly identical frames.

Press `r` to re-register the probe or `q` to cancel.

## Output convention

The output uses the schema consumed directly by `utils.pose_consensus.PoseConsensus.from_yaml()`:

```text
world_frame = <reference camera optical frame>
world_T_camera = reference_T_camera
```

The reference camera is identity. For another camera, `x`, `y`, and `z` are the coordinates of that camera origin expressed in the reference camera optical frame. Orientation is also the camera frame orientation relative to the reference frame.

## Quality checks

The output stores per-camera translation and rotation RMSE. Inspect these values before using consensus. Large errors usually indicate tracking loss, poor probe visibility, unsynchronized motion, inaccurate probe geometry, or insufficient pose diversity.

Because the current multi-camera capture waits sequentially for frames, move the probe slowly. Fast motion increases inter-camera timestamp error. For best calibration, pause briefly at each distinct pose.
