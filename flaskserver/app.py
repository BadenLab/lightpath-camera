"""
Flask camera server for Raspberry Pi.
Provides MJPEG streaming and raw capture endpoints.
"""

import io
import json
import subprocess
import threading
import time
import flask
import camera
import cv as flaskcv
import logging
from pathlib import Path
import functools


app = flask.Flask(__name__)

# Homography detection state (lives in app.py)
_homography_lock = threading.Lock()
_homography_result = None  # {matrix, decomposed, corners, timestamp, found}
_detection_enabled = False
_detection_interval_s = 2.0
_detection_thread = None
_detection_stop_event = threading.Event()
_cached_corners = None  # For overlay drawing


# RANSAC reprojection threshold for the homography fit, in photo pixels.
# It gets converted to mm (the units the fit runs in) via photo_mm_per_pixel.
RANSAC_REPROJ_THRESHOLD_PX = 3.0


def proj_mm_per_pixel():
    # DMD
    # DMD_4500_H = 6161.4; # mm
    # x_axis_mirrors = 912 * 2
    dmd_4500_width_um = 9855
    dmd_4500_width_mm = dmd_4500_width_um / 1000
    return dmd_4500_width_mm / 1280


def photo_mm_per_pixel():
    sensor_H_um, sensor_W_um = camera.sensor_size_um()
    # The Video mode is 720p, which is 16:9, which will be wider than the camera
    # sensor's aspect ratio. Therefore, assume that the width is fully used.
    return (sensor_W_um / 1000) / 1280


def points_to_mm(dmd_points, photo_points):
    proj_points = dmd_points * proj_mm_per_pixel()
    photo_points_mm = photo_points * photo_mm_per_pixel()
    return proj_points, photo_points_mm


def stim_center_mm():
    """Centre of the 1280x800 stimulus frame (= the black marker dot), in mm."""
    scale = proj_mm_per_pixel()
    return (639.5 * scale, 399.5 * scale)


def photo_center_mm():
    """Centre of the 1280x720 camera frame (= the crosshair), in mm."""
    scale = photo_mm_per_pixel()
    return (639.5 * scale, 359.5 * scale)


@functools.lru_cache(maxsize=1)
def checkerboard_src_points():
    """Source points, computed analytically from the pattern geometry.

    Must match the pattern served to the projector
    (static/checkerboard-11r-17c-50s.png, linked from index.html). To catch
    a mismatch between that file and the parameters here, the analytic
    corners are checked once against corners detected in the file.
    """
    corners = flaskcv.checkerboard_corners(
        n_rows=11, n_cols=17, square_size=50, h=800, w=1280
    )
    src_path = Path(app.root_path) / "static" / "checkerboard-11r-17c-50s.png"
    detected = flaskcv.corners_from_file(src_path)
    err_px = flaskcv.max_corner_error(corners, detected)
    assert err_px < 0.5, (
        f"Analytic corners deviate from those detected in {src_path.name} "
        f"by up to {err_px:.3f} px"
    )
    logging.info(f"Analytic corners match {src_path.name} to {err_px:.3f} px")
    return corners


def _detection_loop():
    """Background thread: periodically check for checkerboard."""
    global _homography_result, _cached_corners
    src_pts = checkerboard_src_points()

    while not _detection_stop_event.is_set():
        if not _detection_enabled:
            time.sleep(0.5)
            continue

        try:
            _detect_once(src_pts)
        except Exception:
            logging.exception("Homography detection iteration failed")

        _detection_stop_event.wait(timeout=_detection_interval_s)


def _detect_once(src_pts):
    """Run one detection iteration and update the shared homography state."""
    global _homography_result, _cached_corners

    frame = camera.get_current_frame()
    if frame is None:
        return
    if frame.shape != (720, 1280, 3):
        logging.warning(
            f"Unexpected frame shape: {frame.shape}, expected (720, 1280, 3)"
        )
        return

    corners = flaskcv.detect_corners(frame)
    sharp = flaskcv.sharpness(frame)

    with _homography_lock:
        if corners is not None:
            dst_pts = corners.reshape(-1, 2)
            src_pts_mm, dst_pts_mm = points_to_mm(src_pts, dst_pts)
            threshold_mm = RANSAC_REPROJ_THRESHOLD_PX * photo_mm_per_pixel()
            H = flaskcv.calc_homography(src_pts_mm, dst_pts_mm, threshold_mm)
            if H is not None:
                decomposed = flaskcv.decompose_homography(
                    H, src_center=stim_center_mm(), dst_center=photo_center_mm()
                )
                _homography_result = {
                    "found": True,
                    "timestamp": time.time(),
                    "matrix": H.tolist(),
                    "decomposed": {
                        "rotation_deg": float(decomposed["rotation_deg"]),
                        "zoom": float(decomposed["zoom"]),
                        "scale_x": float(decomposed["scale_x"]),
                        "scale_y": float(decomposed["scale_y"]),
                        "aspect_ratio": float(decomposed["aspect_ratio"]),
                        "perspective_magnitude": float(
                            decomposed["perspective_magnitude"]
                        ),
                        "perspective_direction_deg": float(
                            decomposed["perspective_direction_deg"]
                        ),
                        "center_dx_mm": float(decomposed["center_dx"]),
                        "center_dy_mm": float(decomposed["center_dy"]),
                        "center_dx_px": float(
                            decomposed["center_dx"] / photo_mm_per_pixel()
                        ),
                        "center_dy_px": float(
                            decomposed["center_dy"] / photo_mm_per_pixel()
                        ),
                    },
                    "sharpness": sharp,
                }
                _cached_corners = corners
        else:
            _homography_result = {
                "found": False,
                "timestamp": time.time(),
                "sharpness": sharp,
            }
            _cached_corners = None


def _get_cached_corners():
    """Callback for generate_frames to get current corners."""
    with _homography_lock:
        return _cached_corners


def start_detection(interval_s=2.0):
    global _detection_thread, _detection_enabled, _detection_interval_s
    _detection_interval_s = interval_s
    _detection_enabled = True
    if _detection_thread is None or not _detection_thread.is_alive():
        _detection_stop_event.clear()
        _detection_thread = threading.Thread(
            target=_detection_loop, daemon=True
        )
        _detection_thread.start()


def stop_detection():
    global _detection_enabled, _cached_corners
    _detection_enabled = False
    _detection_stop_event.set()
    with _homography_lock:
        _cached_corners = None


@app.route("/")
def index():
    """Serve the index page with video stream."""
    return flask.render_template("index.html")


@app.route("/stream")
def stream():
    """MJPEG video stream endpoint."""
    return flask.Response(
        camera.generate_frames(get_corners_fn=_get_cached_corners),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/capture", methods=["POST"])
def do_capture():
    """
    Capture a high-resolution photo.

    Query parameters:
        format: "jpeg", "dng" or "raw" (default: "raw")

    Returns:
        The captured image as a file download.
    """
    output_format = flask.request.args.get("format", "raw").lower()

    supported = ["jpeg", "dng", "raw"]
    if output_format not in supported:
        return (
            flask.jsonify(
                {"error": "Invalid format. Use one of: " + ", ".join(supported)}
            ),
            400,
        )

    result = camera.capture(output_format)

    if result is None:
        return flask.jsonify({"error": "Capture failed"}), 500

    res_bytes, metadata = result

    if output_format == "raw":
        data = res_bytes
        mimetype = "application/octet-stream"
        filename = "capture.raw"
    elif output_format == "dng":
        data = res_bytes
        mimetype = "image/x-adobe-dng"
        filename = "capture.dng"
    elif output_format in ("jpeg", "jpg"):
        data = res_bytes
        mimetype = "image/jpeg"
        filename = "capture.jpg"
    else:
        assert False, "Unreachable"

    return flask.send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@app.route("/last", methods=["GET"])
def last_capture():
    """
    Return the last captured photo.

    Query parameters:
        format: "jpeg" or "dng" (default: "dng")

    Returns:
        The last captured image, or 404 if none exists.
    """
    result = camera.get_last_capture()

    if result is None:
        return flask.jsonify({"error": "No capture available"}), 404

    jpeg_bytes, dng_bytes, metadata = result
    output_format = flask.request.args.get("format", "dng").lower()

    if output_format == "dng":
        data = dng_bytes
        mimetype = "image/x-adobe-dng"
        filename = "last_capture.dng"
    else:
        data = jpeg_bytes
        mimetype = "image/jpeg"
        filename = "last_capture.jpg"

    return flask.send_file(
        io.BytesIO(data),
        mimetype=mimetype,
        as_attachment=True,
        download_name=filename,
    )


@app.route("/last/info", methods=["GET"])
def last_capture_info():
    """Return metadata about the last capture without the image data."""
    result = camera.get_last_capture()

    if result is None:
        return flask.jsonify({"error": "No capture available"}), 404

    jpeg_bytes, dng_bytes, metadata = result

    return flask.jsonify(
        {
            "exposure_time": metadata.get("ExposureTime"),
            "analogue_gain": metadata.get("AnalogueGain"),
            "jpeg_size": len(jpeg_bytes),
            "dng_size": len(dng_bytes),
        }
    )


@app.route("/settings", methods=["GET"])
def get_settings():
    """Get current camera settings."""
    return flask.jsonify(camera.get_settings())


@app.route("/config", methods=["GET"])
def get_config():
    """Get current camera configuration."""
    return flask.jsonify({"config_str": str(camera.get_config())})


@app.route("/settings", methods=["POST"])
def set_settings():
    """
    Set camera settings.

    JSON body:
        exposure_time: Exposure time in miliseconds (1-100,000)
        analogue_gain: Analogue gain (1.0-16.0), optional

    This disables auto exposure and sets manual focus mode.
    """
    data = flask.request.get_json()

    if data is None:
        return flask.jsonify({"error": "JSON body required"}), 400

    exposure_time = data.get("exposure_time")
    analogue_gain = data.get("analogue_gain")

    if exposure_time is None:
        return flask.jsonify({"error": "exposure_time required"}), 400

    try:
        if exposure_time < 1 or exposure_time > 100_000:
            return (
                flask.jsonify(
                    {"error": "exposure_time must be 1 ms and 100 s"}
                ),
                400,
            )
    except (ValueError, TypeError):
        return flask.jsonify({"error": "exposure_time must be an integer"}), 400

    if analogue_gain is not None:
        try:
            analogue_gain = float(analogue_gain)
            if analogue_gain < 1.0 or analogue_gain > 16.0:
                return (
                    flask.jsonify({"error": "analogue_gain must be 1.0-16.0"}),
                    400,
                )
        except (ValueError, TypeError):
            return (
                flask.jsonify({"error": "analogue_gain must be a number"}),
                400,
            )

    camera.set_exposure(exposure_time, analogue_gain)

    return flask.jsonify(camera.get_settings())


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    Shut down the Raspberry Pi.

    The shutdown is delayed slightly so that the HTTP response can be
    delivered before the network goes down.
    """

    def do_shutdown():
        subprocess.run(["sudo", "shutdown", "-h", "now"])

    threading.Timer(1.0, do_shutdown).start()
    return flask.jsonify({"status": "shutting down"})


@app.route("/homography", methods=["GET"])
def get_homography_route():
    """Get current homography result."""
    with _homography_lock:
        result = _homography_result
    return flask.jsonify(result or {"found": False})


@app.route("/homography/events")
def homography_events():
    """SSE stream for homography updates."""

    def generate():
        last_timestamp = None
        while True:
            with _homography_lock:
                result = _homography_result
            if result and result.get("timestamp") != last_timestamp:
                last_timestamp = result.get("timestamp")
                yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.5)

    return flask.Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.route("/homography/start", methods=["POST"])
def start_detection_route():
    """Start homography detection."""
    data = flask.request.get_json() or {}
    interval = data.get("interval", 2.0)
    start_detection(interval_s=interval)
    return flask.jsonify({"status": "started", "interval": interval})


@app.route("/homography/stop", methods=["POST"])
def stop_detection_route():
    """Stop homography detection."""
    stop_detection()
    return flask.jsonify({"status": "stopped"})


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Fail fast if the analytic corners don't match the pattern file.
    checkerboard_src_points()
    print("Starting camera server...")
    print("Endpoints:")
    print("  GET  /          - Index page with stream viewer")
    print("  GET  /stream    - MJPEG video stream")
    print("  POST /capture   - Capture photo (?format=jpeg|dng)")
    print("  GET  /last      - Get last captured photo")
    print("  GET  /settings  - Get current settings")
    print("  POST /settings  - Set exposure time")
    print()

    # Run on all interfaces so it's accessible from other machines
    app.run(host="0.0.0.0", port=5000, threaded=True)
