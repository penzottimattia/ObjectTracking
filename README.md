# ObjectTracking

Real-time 6-DoF object pose tracking using **SAM3** for automatic detection and **FoundationPose** for pose estimation. Works standalone (console output) or with ROS (`PoseStamped` publishing).

## How It Works

1. You provide an object name (e.g., `--object cup`) that matches a folder in `object/`
2. **SAM3** uses the object name as a text prompt to detect and segment the object in the camera feed
3. **FoundationPose** computes the initial 6-DoF pose from the segmentation mask
4. Subsequent frames use FoundationPose's fast tracking mode
5. Pose is either printed to console (`track_object.py`) or published via ROS (`track_object_ros.py`)
6. Periodic SAM3 re-detection recovers from tracking drift

## Project Structure

```
ObjectTracking/
├── object/                  # Object meshes (.obj + .mtl per object)
│   ├── cup/
│   ├── can/
│   └── ...
├── FoundationPose/          # 6-DoF pose estimation (NVIDIA)
│   ├── estimater.py         # Core FoundationPose class
│   ├── weights/             # Pretrained model weights
│   └── ...
├── sam3/                    # Segment Anything Model 3 (Facebook)
├── scripts/
│   ├── tracking_utils.py    # Shared: SAM3, FP, camera, vis helpers
│   ├── track_object.py      # Standalone tracker (no ROS)
│   └── track_object_ros.py  # ROS tracker (publishes PoseStamped)
└── tests/
```

## Prerequisites

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.x driver (tested: RTX 4070 / 5090, driver 575+)
- Intel RealSense D435 camera
- ROS Noetic (only for `track_object_ros.py`, not needed for standalone)

## Installation

Everything runs in a **single conda environment**.

### Step 1: Install Intel RealSense SDK (system-level)

This lets your system see the D435. Run these commands:

```bash
# Install dependencies
sudo apt-get update && sudo apt-get install -y \
    libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev cmake

# Add Intel RealSense apt repo
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | \
    sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] \
    https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
    sudo tee /etc/apt/sources.list.d/librealsense.list

sudo apt-get update

# Install the SDK
sudo apt-get install -y librealsense2-dkms librealsense2-utils
```

**Verify**: Plug in the D435 via USB 3.0, then run:

```bash
realsense-viewer
```

You should see the RGB and depth streams from your camera. Close the viewer before continuing.

### Step 2: Create conda environment

```bash
conda create -n objtrack python=3.9 -y
conda activate objtrack
```

### Step 3: Install PyTorch (CUDA 12.4)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **Note on GPU compatibility**: PyTorch CUDA 12.4 wheels work on any NVIDIA driver
> that supports CUDA 12.x (driver 525+). This includes RTX 4070, 5090, and all
> other modern GPUs. The Python version (3.9) is independent of the GPU -- GPUs
> only care about the CUDA driver, not Python.

### Step 4: Install FoundationPose dependencies

```bash
cd FoundationPose

# Install Eigen3 (needed to build C++ extensions)
conda install conda-forge::eigen=3.4.0 -y

# Install Python dependencies
pip install -r requirements.txt

# Install NVDiffRast (GPU rasterizer)
pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install PyTorch3D (build from source -- takes a few minutes)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# Build FoundationPose C++ extensions
export CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11:$CMAKE_PREFIX_PATH"
bash build_all_conda.sh

cd ..
```

### Step 5: Install SAM3

```bash
# SAM3 checkpoints are on Hugging Face (requires access approval)
pip install huggingface_hub
huggingface-cli login
# Go to https://huggingface.co/facebook/sam3 and request access first

# Install SAM3
cd sam3
pip install -e .
cd ..
```

### Step 6: Install remaining tools

```bash
pip install pyrealsense2 scipy
```

### Step 7: Download FoundationPose weights

Download from the [FoundationPose Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i):

```
FoundationPose/weights/
├── 2023-10-28-18-33-37/     # refiner weights
└── 2024-01-11-20-02-45/     # scorer weights
```

### Step 8: Verify setup

```bash
# Check that the camera is detected
python -c "import pyrealsense2 as rs; ctx = rs.context(); print(f'Found {len(ctx.devices)} device(s)'); [print(f'  {d.get_info(rs.camera_info.name)}') for d in ctx.devices]"

# Check that FoundationPose loads
python -c "import sys; sys.path.insert(0, 'FoundationPose'); from estimater import FoundationPose; print('FoundationPose OK')"

# Check that SAM3 loads
python -c "from sam3 import build_sam3_image_model; print('SAM3 OK')"
```

## Camera Intrinsics

**You do NOT need to manually configure camera intrinsics.** The scripts automatically read the D435's factory-calibrated intrinsics (fx, fy, cx, cy) via `pyrealsense2` at startup. Depth is also automatically aligned to the color frame. No ROS camera node is needed -- the scripts talk to the camera directly.

## Adding Your Own Objects

Place your object's mesh files in `object/<name>/`:

```
object/my_object/
├── my_object.obj
├── my_object.mtl
└── texture_map.png   # optional
```

The `--object` argument must match the folder name. SAM3 uses this same name as the text prompt for detection, so use a descriptive name (e.g., `cup` not `obj1`).

## Usage

### Standalone (no ROS)

```bash
conda activate objtrack

# Basic tracking -- prints pose to console, shows visualization
python scripts/track_object.py --object cup

# Lower detection confidence for harder objects
python scripts/track_object.py --object can --confidence 0.3

# Headless mode (no OpenCV window)
python scripts/track_object.py --object cup --no-vis

# Disable image flip (default flips for upside-down mounted cameras)
python scripts/track_object.py --object cup --no-flip
```

### With ROS

Requires ROS Noetic. In a separate terminal, start `roscore` first.

```bash
conda activate objtrack

# Publish to /object_pose
python scripts/track_object_ros.py --object cup

# Custom topic and confidence
python scripts/track_object_ros.py --object can --topic /can_pose --confidence 0.4

# Adjust re-detection interval (seconds, 0 = disabled)
python scripts/track_object_ros.py --object cup --redetect_interval 10
```

### Keyboard Controls (when visualization is enabled)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `r` | Reset tracking (triggers SAM3 re-detection) |

### ROS Topic Output

| Topic | Type | Description |
|-------|------|-------------|
| `/object_pose` | `geometry_msgs/PoseStamped` | Object pose in camera optical frame |

The pose contains:
- `position`: (x, y, z) in meters, in camera optical frame
- `orientation`: quaternion (x, y, z, w)
- `header.frame_id`: `camera_color_optical_frame` (default)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No .obj file found` | Check `object/<name>/` has a `.obj` file |
| `Found 0 device(s)` | Plug D435 into a **USB 3.0** port (blue port). Try `realsense-viewer` first |
| SAM3 not detecting | Lower `--confidence` (try `0.2`), or use a more descriptive object name |
| Slow FPS | Reduce `--track_refine_iter 1` for speed at some accuracy cost |
| CUDA out of memory | Close other GPU apps. Both models together need ~8-10 GB VRAM |
| `pyrealsense2` import error | Run `pip install pyrealsense2` or check the SDK installed in Step 1 |
| FoundationPose C++ build fails | Check `conda list eigen` shows 3.4.0, and that `CMAKE_PREFIX_PATH` is set |

## License

- FoundationPose: [NVIDIA Source Code License](FoundationPose/LICENSE)
- SAM3: See `sam3/LICENSE`
