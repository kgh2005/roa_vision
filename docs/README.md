# ROA Vision

ROS 2 vision workspace for the RO:BIT robot. It combines ZED Mini RGB/depth data with TensorRT object detection to publish RoboCup vision data and control the camera pan-tilt unit.

## Packages

| Package | Purpose |
| --- | --- |
| `vision_interfaces` | Custom ROS 2 vision messages |
| `vision_tensorrt` | TensorRT-based YOLO detector and ZED launch |
| `roa_vision_pipeline` | Detection refinement and ball tracking |
| `roa_vision_description` | Camera URDF, TF, and RViz configuration |

## Requirements

- Ubuntu 24.04 and ROS 2 Jazzy
- CUDA Toolkit and TensorRT
- ZED SDK and `zed_wrapper`
- OpenCV and `cv_bridge`
- `dynamixel_rdk_msgs`

The TensorRT engine must be compatible with the target GPU and TensorRT version.

## Makefile

Run the following commands from the repository root:

| Command | Description |
| --- | --- |
| `make install` | Install project-specific apt and Python dependencies |
| `make install-apt` | Install TensorRT and OpenCV development packages |
| `make install-python` | Install Python model-export and controller dependencies |
| `make export` | Export the `.pt` model in `src/vision_tensorrt/model` to TensorRT |
| `make build` | Build the vision packages sequentially from the parent colcon workspace |
| `make help` | Show available Makefile targets |

`make export` expects exactly one `.pt` file in `src/vision_tensorrt/model` and requires the Ultralytics `yolo` command.

`make build` treats `../..` as the colcon workspace root. It builds `vision_interfaces` first, then sources the workspace before building the remaining packages one at a time. Override the detected path when needed with `make build WORKSPACE_DIR=/path/to/workspace`.

## Build

```bash
cd <colcon-workspace>/src/roa_vision
make install
make build
source ../../install/setup.bash
```

## Run

Start the complete pipeline:

```bash
ros2 launch roa_vision_pipeline zedm_vision_pipeline.launch.py
```

This starts the ZED Mini, TensorRT detector, robot/camera TF, detection refiner, and ball tracker.

Debug the TF GUI without starting the ball tracker:

```bash
ros2 launch roa_vision_pipeline tf_gui_debug.launch.py
```

This starts the ZED Mini, TensorRT detector, robot/camera TF GUI, and detection refiner. It does not start the ball tracker.

Start only the ZED camera and TensorRT detector:

```bash
ros2 launch vision_tensorrt vision_tensorrt.launch.py
```

## Detection output

Detections are published on `/Bounding_box` using `vision_interfaces/msg/BoundingBox`:

```text
int32[] class_ids
float32[] score
int32[] x1
int32[] y1
int32[] x2
int32[] y2
```

All arrays have the same length. Values at the same index describe one detection. The default class mapping is `0: ball`, `1: robot`, `2: L`, `3: T`, `4: X`, and `5: goal`.

## Configuration

- TensorRT detector: `src/vision_tensorrt/config/params.yaml`
- Detection refiner and ball tracker: `src/roa_vision_pipeline/config/vision_pipeline.yaml`
- TensorRT model: `src/vision_tensorrt/model/best.engine`
