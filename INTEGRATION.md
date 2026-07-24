# Optional multi-camera pose consensus

The submodule is deliberately ROS-independent. Both tracking entry points use the same `PoseConsensus` and `evaluate_tracked_objects()` API.

## Pose convention

For each camera, the configuration must contain `world_T_camera`. FoundationPose supplies `camera_T_object`. The submodule computes:

```text
world_T_object = world_T_camera @ camera_T_object
```

Do not insert `camera_T_world` without inverting it first.

## Shared CLI arguments

Add these arguments to both `scripts/track_object.py` and `scripts/track_object_ros.py`:

```python
parser.add_argument(
    "--consensus-config",
    default=None,
    help="Enable pose consensus using world_T_camera transforms from this YAML file",
)
parser.add_argument("--consensus-translation", type=float, default=None)
parser.add_argument("--consensus-rotation-deg", type=float, default=None)
parser.add_argument("--consensus-failures", type=int, default=None)
```

After `cameras` has been parsed, initialize the optional controller:

```python
consensus = None
if args.consensus_config:
    from utils.pose_consensus import PoseConsensus
    consensus = PoseConsensus.from_yaml(
        args.consensus_config,
        translation_tolerance_m=args.consensus_translation,
        rotation_tolerance_deg=args.consensus_rotation_deg,
        failures_before_reset=args.consensus_failures,
    )
```

The `from_yaml()` method ignores `None` overrides and retains values from the YAML file.

## Shared tracking-loop integration

Immediately after `track_batch(...)` updates every camera state:

```python
consensus_results = {}
if consensus is not None:
    from utils.consensus_tracking import evaluate_tracked_objects, should_reset_any

    consensus_results = evaluate_tracked_objects(consensus, tracked, cameras)
    if should_reset_any(consensus_results):
        logging.warning("Pose consensus failed; resetting every camera/object tracker")
        reset()
        continue
```

Resetting all camera/object states is intentional. A failed consensus cannot reliably identify which camera is wrong when only two cameras are available. With three or more cameras, an outlier rejection policy can be added later.

In the standalone script, print or display the accepted common-frame pose:

```python
for object_name, result in consensus_results.items():
    if result.consistent and result.pose_world is not None:
        print_pose(result.pose_world, f"{object_name}@{consensus.world_frame}")
```

## ROS publishing

Create one fused publisher per object after ROS node initialization:

```python
from geometry_msgs.msg import PoseStamped

fused_publishers = {
    obj["name"]: node.create_publisher(
        PoseStamped,
        f"/object_pose_fused/{obj['name']}",
        10,
    )
    for obj in tracked
} if consensus is not None else {}
```

After consensus evaluation, publish only consistent results:

```python
for object_name, result in consensus_results.items():
    if result.consistent and result.pose_world is not None:
        fused_publishers[object_name].publish(
            pose_msg(result.pose_world, consensus.world_frame)
        )
```

If the tracker already has an internal `reset()` function, call it directly. This is safer and faster than having the node call its own `/reset_tracker` service. The existing reset service remains available to external clients and should call the same `reset()` function.

## Recommended behavior

- Require at least two initialized camera measurements.
- Compare both translation and orientation.
- Require several consecutive failures before resetting to avoid reacting to one bad frame.
- Do not publish a fused pose when consensus fails.
- Average translation arithmetically and rotations with quaternion eigenvector averaging.
- Reset all trackers after a confirmed disagreement when only two cameras are present.

## Example

```bash
python scripts/track_object_ros.py \
  --object probe \
  --cameras robotcam tablecam \
  --consensus-config config/pose_consensus.yaml
```
