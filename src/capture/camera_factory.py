from typing import Any, Dict, Tuple

from capture.camera_gst import CameraGST


def _active_camera_set(hw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    active_name = str(hw_cfg.get("active_camera_set", "b0429_single")).strip()
    camera_sets = hw_cfg.get("camera_sets", {}) or {}
    camera_set = camera_sets.get(active_name)

    if not isinstance(camera_set, dict):
        raise ValueError(f"active camera_set not found: {active_name}")

    return camera_set


def _primary_camera_cfg(camera_set: Dict[str, Any]) -> Dict[str, Any]:
    cameras = camera_set.get("cameras", []) or []
    if not cameras:
        raise ValueError("camera_set.cameras is empty")

    cam_cfg = cameras[0]
    if not isinstance(cam_cfg, dict):
        raise ValueError("camera config must be object")

    return cam_cfg


def build_gst_pipeline(cam_cfg: Dict[str, Any]) -> str:
    manual_pipeline = str(cam_cfg.get("gst_pipeline", "") or "").strip()
    if manual_pipeline:
        return manual_pipeline

    pipeline_type = str(cam_cfg.get("pipeline_type", "b0429_gray16_to_gray8")).strip()
    device = str(cam_cfg.get("device", "/dev/video0"))
    width = int(cam_cfg.get("width", 1280))
    height = int(cam_cfg.get("height", 720))
    fps = int(cam_cfg.get("fps", 30))

    if pipeline_type == "b0429_gray16_to_gray8":
        input_format = str(cam_cfg.get("input_format", "GRAY16_LE"))
        output_format = str(cam_cfg.get("output_format", "GRAY8"))
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,format={input_format},width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! video/x-raw,format={output_format} ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    if pipeline_type == "v4l2_bgr":
        return (
            f"v4l2src device={device} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    if pipeline_type == "nvargus_bgr":
        sensor_id = int(cam_cfg.get("sensor_id", 0))

        # Output size used by the application
        out_width = width
        out_height = height
        out_fps = fps

        # Native Argus capture mode.
        # IMX477 does not expose 1920x1080@30 as native mode.
        # Use 3840x2160@30, then scale down to 1920x1080 for OpenCV.
        cap_width = int(cam_cfg.get("capture_width", out_width))
        cap_height = int(cam_cfg.get("capture_height", out_height))
        cap_fps = int(cam_cfg.get("capture_fps", out_fps))

        flip_method = int(cam_cfg.get("flip_method", 0))

        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={cap_width},height={cap_height},framerate={cap_fps}/1,format=NV12 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw,width={out_width},height={out_height},format=BGRx ! "
            "videorate ! "
            f"video/x-raw,format=BGRx,framerate={out_fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    raise ValueError(f"unsupported pipeline_type: {pipeline_type}")


def create_camera_from_hardware_config(hw_cfg: Dict[str, Any]) -> Tuple[CameraGST, Dict[str, Any]]:
    camera_set = _active_camera_set(hw_cfg)
    cam_cfg = _primary_camera_cfg(camera_set)
    gst = build_gst_pipeline(cam_cfg)

    cam = CameraGST(gst)

    info = dict(cam_cfg)
    info["gst_pipeline"] = gst
    info["camera_set_mode"] = str(camera_set.get("mode", "single"))
    info["camera_set_backend"] = str(camera_set.get("backend", "gstreamer"))

    return cam, info