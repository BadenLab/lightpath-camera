"""
Camera module using picamera2.
Provides streaming and capture functions.
"""

import io
import threading
from typing import Optional, Literal
import numpy as np
import picamera2 as pc2
import picamera2.encoders
import picamera2.outputs
import libcamera
import PIL.Image
import PIL.ImageDraw


import logging

_logger = logging.getLogger(__name__)


class StreamingOutput(io.BufferedIOBase):
    """Output buffer for MJPEG streaming with synchronization."""

    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


# Module state
_camera: pc2.Picamera2 = None
_streaming = False
_stream_output: StreamingOutput = None
_last_capture = None  # (jpeg_bytes, dng_bytes, metadata)

# Settings
_exposure_time_us = 100000  # 100ms default
_analogue_gain = 1.0


def default_controls():
    res = {
        "AeEnable": False,
        "ExposureTime": _exposure_time_us,
        "AnalogueGain": _analogue_gain,
        "AwbEnable": False,
        "NoiseReductionMode": libcamera.controls.draft.NoiseReductionModeEnum.Fast,
        "FrameDurationLimits": (100, 100_000_000),
        # "NoiseReductionMode": libcamera.controls.draft.NoiseReductionModeEnum.Off,
        # "FrameDurationLimits": (33333, 33333),
        # "ScalerCrop": (0, 0, 1280, 720),
    }
    if has_afmode():
        res["AfMode"] = libcamera.controls.AfModeEnum.Manual
    return res


def align_stream(stream_config) -> None:
    """
    This function is copied from Picamera2.Picamera2.align_stream().

    What does it do?

    Rows will be padded to 64-bit boundaries. Dealing with that can be a pain,
    so we choose to re-size the image (downward) to avoid it using align_stream().
    This is about updating the size so that all planes are multiples of 64bits,
    which is necessary for the Raspberry Pi's hardware.

    If you peak into Picamera2.align_stream(), you will see special treatment for
    YUV formats. This is because all three planes need to be aligned, that is the
    Y (luma) plane) and the two chroma planes (U and V). The latter two have rows
    that are half as long as the Y plane (by design). In effect, if Y is of length
    W, then both W and W/2 must be multiples of 64 bits. This can be done by
    aligning W to 128 bits.

    There is also special treatment for some of the RGB formats, such as XBGR8888.
    These formats already use 4 bytes per pixel, meaning alignment to 64 bits
    is easier and is done by aligning pixels to multiples of 16. Thus, Picamera2's
    align_stream() reduces the pixel aligment boundary.

    So far, we have only considered rows, which implicate the width of the image.
    The height is also reduced to a multiple of 2, as some formats (e.g. YUV420)
    require even heights.
    """
    # Adjust the image size so that all planes are a multiple of 32/64 bytes wide.
    # This matches the hardware behaviour and means we can be more efficient.
    align = 32 if pc2.Picamera2.platform == pc2.platform.Platform.VC4 else 64
    if stream_config["format"] in ("YUV420", "YVU420"):
        align *= 2  # because the UV planes will have half this alignment
    elif stream_config["format"] in (
        "XBGR8888",
        "XRGB8888",
        "RGB161616",
        "BGR161616",
    ):
        # note by me: why not divide by 4?
        align //= 2  # we have an automatic extra factor of 2 here
    size = stream_config["size"]
    stream_config["size"] = (size[0] - size[0] % align, size[1] - size[1] % 2)
    return stream_config


def create_config(use_lores, do_vflip):
    assert _camera is not None
    # According to this response by a maintainer: https://github.com/raspberrypi/picamera2/discussions/585#discussioncomment-5165945
    # the size request describes the output size, and the camera will try to use as many
    # pixels as possible so long as the aspect ratio is maintained.
    video_size = (1280, 720)  # 720p
    main_config = {
        # Endianess means that this will be received as RGBX.
        "format": "XBGR8888",
        "size": video_size,
    }
    lores_config = {
        "format": "YUV420",
        "size": video_size,
    }
    if has_preserve_ar():
        # Keep aspect ratio by cropping if needed.
        main_config["preserve_ar"] = True
        lores_config["preserve_ar"] = True
    main_config = align_stream(main_config)
    lores_config = align_stream(lores_config)
    # This line causes sudden video degradation!
    # high_res_mode = _camera.sensor_modes[0]  # see p21 in pdf docs.
    high_res_mode = _camera.sensor_modes[-1]  # see p21 in pdf docs.
    # high_res_mode = {
    #     "size": (4656, 3496),
    #     "bit_depth": 10,
    # }
    # assert high_res_mode["size"] == (4656, 3496)
    # The Bayer pattern changes on flip:
    # RG    ->     GB
    # GB           RG
    format = "SGBRG10_CSI2P" if do_vflip else "SRGGB10_CSI2P"
    config = {
        # Not sure if needed to be set, as they are just for picamea2
        "use_case": "video_and_raw",
        "display": "lores" if use_lores else "main",
        # encode is the stream to be encoded as a video stream.
        "encode": "lores" if use_lores else "main",
        "transform": libcamera.Transform(vflip=do_vflip),
        "colour_space": libcamera.ColorSpace.Rec709(),
        # "colour_space": libcamera.ColorSpace.Smpte170m(),
        # "colour_space": None,
        "buffer_count": 6,
        "queue": True,
        "main": main_config,
        "lores": lores_config if use_lores else None,
        "raw": {
            "format": format,
            "size": high_res_mode["size"],
            # If we want unpacked:
            # "format": "SRGGB10",
            # there isn't any size option, as it's determined by the sensor config.
        },
        "controls": default_controls(),
        "sensor": {
            "output_size": high_res_mode["size"],
            "bit_depth": high_res_mode["bit_depth"],
        },
    }
    _logger.info(f"Scaler crop: {_camera.camera_controls['ScalerCrop'][2]}")
    return config


def camera_model():
    res = _camera.camera_properties["Model"]
    return res


def has_afmode():
    res = ({"ov5647": False}).get(camera_model(), True)
    return res


def has_preserve_ar():
    res = ({"ov5647": False}).get(camera_model(), True)
    return res


def pixel_size_um():
    """
    imx519: https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/16MP-IMX519/
    ov5647: https://docs.arducam.com/Raspberry-Pi-Camera/Native-camera/5MP-OV5647/

    imx519 has a diagonal of 7.103 mm.
    """
    res = {"imx519": 1.22, "ov5647": 1.4}.get(camera_model(), None)
    return res


def sensor_size_um():
    """
    ov5647: https://cdn.sparkfun.com/datasheets/Dev/RaspberryPi/ov5647_full.pdf

    Return:
        (H_um, W_um) in micrometers, or None if unknown.
    """
    model = camera_model()
    if model == "imx519":
        # diag = 7.103
        # sensor_size = diag / sqrt(4656^2 + 3496^2) = 0.001219943
        sensor_size = 1.219943
        H_um = sensor_size * 3496
        W_um = sensor_size * 4656
    elif model == "ov5647":
        H_um = 2740
        W_um = 3670
    else:
        return None, None
    return (H_um, W_um)


def init_camera():
    global _camera
    if _camera is not None:
        return
    _camera = pc2.Picamera2()
    _logger.info(f"Camera initialized: {_camera.camera_properties['Model']}")
    _logger.debug(f"Camera properties: {_camera.camera_properties}")
    # Our canonical reference frame is the person standing over the setup,
    # looking down at the projection target. The viewer's top-left is (0, 0).
    # The camera sensor's default top left is the viewer's bottom-left, and so
    # we will flip the camera's vertical axis to match the viewer's perspective.
    config = create_config(use_lores=False, do_vflip=True)
    _camera.configure(config)


def start_mjpeg_stream():
    global _streaming, _stream_output

    if _streaming:
        return

    init_camera()

    _stream_output = StreamingOutput()
    # JpegEncoder is a software encoder, multithreaded.
    encoder = pc2.encoders.JpegEncoder(q=70)
    # The MJPEG encoder doesn't seem to work well, and it sort of crashes if I try
    # to have the max resolution raw sensor resolution.
    # encoder = pc2.encoders.MJPEGEncoder()
    output = pc2.outputs.FileOutput(_stream_output)
    _camera.start_recording(encoder, output, quality=pc2.encoders.Quality.LOW)

    _streaming = True
    print("Streaming started")


def stop_stream():
    """Stop streaming."""
    global _streaming, _stream_output

    if not _streaming:
        return

    # _camera.stop_recording()
    # stop_recording is just:
    #   stop() # stop the camera
    #   stop_encoder() # stop the encoder
    # But we only want to stop the encoder.
    _camera.stop_encoder()
    _stream_output = None
    _streaming = False
    print("Streaming stopped")


def add_overlay(frame: bytes, corners=None) -> bytes:
    """Add overlays to the JPEG frame. Optionally draw checkerboard corners."""

    img = PIL.Image.open(io.BytesIO(frame))
    draw = PIL.ImageDraw.Draw(img)
    w, h = img.size
    # Crosshair midlines
    draw.line([(w // 2, 0), (w // 2, h)], fill="#707070", width=2)
    draw.line([(0, h // 2), (w, h // 2)], fill="#707070", width=2)
    # Draw corners if provided
    if corners is not None:
        for pt in corners.reshape(-1, 2):
            x, y = int(pt[0]), int(pt[1])
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill="#00ff00")

    out_io = io.BytesIO()
    img.save(out_io, format="JPEG", quality=90)
    return out_io.getvalue()


def generate_frames(get_corners_fn=None):
    """Generator yielding JPEG frames for MJPEG streaming.

    Args:
        get_corners_fn: Optional callback that returns current corners for overlay.
    """
    start_mjpeg_stream()

    while _streaming:
        with _stream_output.condition:
            _stream_output.condition.wait(timeout=1.0)
            frame = _stream_output.frame
            if frame is None:
                continue
            corners = get_corners_fn() if get_corners_fn else None
            frame = add_overlay(frame, corners=corners)

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


def capture(output_format: Literal["raw", "dng", "jpeg"] = "raw"):
    """
    Capture high-resolution JPEG and DNG.
    Returns (jpeg_bytes, dng_bytes, metadata) or None on error.
    """
    global _last_capture

    init_camera()

    def do_raw(request):
        raw_config = _camera.camera_config["raw"]
        width, height = raw_config["size"]
        stride = raw_config["stride"]
        fmt = raw_config["format"]
        assert fmt in (
            "SRGGB10_CSI2P",
            "SGBRG10_CSI2P",
        ), "Only supports packed 10-bit raw"
        res = bytes(request.make_buffer("raw"))
        return res

    def do_dng(request):
        # Create DNG
        # raw_buf = request.make_buffer("raw")
        dng_io = io.BytesIO()
        request.save_dng(dng_io, "raw")
        dng_bytes = dng_io.getvalue()
        return dng_bytes

    def do_jpeg(request):
        jpeg = request.make_image("main")
        jpeg_io = io.BytesIO()
        jpeg.save(jpeg_io, format="JPEG", quality=90)
        jpeg_bytes = jpeg_io.getvalue()
        return jpeg_bytes

    request = None
    try:
        request = _camera.capture_request(flush=True)
        metadata = request.get_metadata()

        if output_format == "raw":
            res = do_raw(request)
        elif output_format == "dng":
            res = do_dng(request)
        elif output_format == "jpeg":
            res = do_jpeg(request)
        else:
            raise ValueError("Invalid output_format")
        _last_capture = (res, metadata)
        return _last_capture

    except Exception as e:
        print(f"Capture error: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if request is not None:
            request.release()


def get_last_capture():
    """Return last capture (jpeg_bytes, dng_bytes, metadata) or None."""
    return _last_capture


def set_exposure(exposure_time_ms: int, analogue_gain: Optional[float] = None):
    """Set manual exposure settings."""
    global _exposure_time_us, _analogue_gain

    _exposure_time_us = exposure_time_ms * 1000
    if analogue_gain is not None:
        _analogue_gain = analogue_gain

    if _camera is not None:
        controls = {
            "AeEnable": False,
            "ExposureTime": _exposure_time_us,
            "AnalogueGain": _analogue_gain,
        }
        if has_afmode():
            controls["AfMode"] = 0  # Manual
        _camera.set_controls(controls)


def get_settings():
    """Return current settings dict."""
    exposure_time_ms = _exposure_time_us // 1000
    res = {
        "exposure_time": exposure_time_ms,
        "analogue_gain": _analogue_gain,
        "AeEnabled": False,
    }
    if has_afmode():
        res["AfMode"] = _camera.camera_controls["AfMode"] if _camera else None
    return res


def get_config():
    """Return current camera configuration."""
    if _camera is None:
        return None
    return _camera.camera_config


def get_current_frame():
    """Get current frame as numpy RGB array for CV processing.

    Grabs a frame directly from the main stream via capture_array, avoiding
    the JPEG round-trip through the MJPEG encoder (no compression artifacts,
    no decode cost). Starts the camera if it isn't running, so this works
    without any streaming client.
    """
    init_camera()
    if not _camera.started:
        _camera.start()
    arr = _camera.capture_array("main")
    # XBGR8888 arrives as RGBX; drop the padding channel.
    return np.ascontiguousarray(arr[:, :, :3])
