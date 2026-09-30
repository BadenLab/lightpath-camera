"""
Computer vision functions.
"""

import cv2
import numpy as np
import PIL.Image
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

CORNERS_PER_ROW = 17 - 1
CORNERS_PER_COL = 11 - 1


def corners_from_file(img_path):
    """Generate source points."""
    assert Path(img_path).exists(), f"Image file not found: {img_path}"
    # convert: detect_corners expects RGB, but the pattern PNG may be grayscale.
    img = PIL.Image.open(img_path).convert("RGB")
    img_arr = np.array(img)
    corners = detect_corners(img_arr)
    assert corners is not None
    return corners


def detect_corners(arr, normalize=True):
    """
    Args:
        arr: (H, W, C) uint8 array in [0, 255], with C = RGB.
    """

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if normalize:
        # Normalize target to use the full [0, 255] range. Captures under
        # narrowband illumination (e.g. single LED) often have low contrast
        # which hurts corner detection.
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    # The markers don't seem to work very well.
    # https://github.com/opencv/opencv/issues/22083
    # CALIB_CB_LARGER is not used: it is for boards extending past the field
    # of view, and it can return more corners than patternSize, which we
    # would have no way to interpret without the meta output.
    is_found, corners = cv2.findChessboardCornersSB(
        gray,
        (CORNERS_PER_ROW, CORNERS_PER_COL),
        flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not is_found:
        return None
    n_expected = CORNERS_PER_ROW * CORNERS_PER_COL
    if len(corners) != n_expected:
        _logger.warning(f"Expected {n_expected} corners, got {len(corners)}")
        return None
    return corners


def checkerboard_corners(n_rows, n_cols, square_size, h, w):
    """Inner-corner coordinates of a centered checkerboard, analytically.

    The pattern (as rasterized by stimlc.calibration.radon_checkerboard) is
    centered in the frame, so its inner corners are known in closed form and
    no detection on the source image is needed.

    Args:
        n_rows, n_cols: Number of checkerboard cells along each axis.
        square_size: Side length of a cell in pixels.
        h, w: Frame height and width in pixels.

    Returns:
        ((n_rows-1)*(n_cols-1), 1, 2) float32 array of (x, y) corners in
        row-major order (matching findChessboardCornersSB), using OpenCV's
        pixel-center convention (integer coordinates are pixel centers, so
        pixel boundaries fall on half-integers).
    """
    x0 = (w - n_cols * square_size) / 2 - 0.5
    y0 = (h - n_rows * square_size) / 2 - 0.5
    xs = x0 + square_size * np.arange(1, n_cols)
    ys = y0 + square_size * np.arange(1, n_rows)
    corners = np.stack([np.tile(xs, len(ys)), np.repeat(ys, len(xs))], axis=-1).astype(
        np.float32
    )
    return corners.reshape(-1, 1, 2)


def max_corner_error(pts_a, pts_b):
    """Max distance between two corner sets, after aligning their ordering."""
    a = pts_a.reshape(-1, 2)
    b = align_corner_ordering(a, pts_b.reshape(-1, 2))
    return float(np.linalg.norm(a - b, axis=1).max())


def draw_corners(arr, corners):
    img_cpy = arr.copy()
    cv2.drawChessboardCorners(
        img_cpy,
        (CORNERS_PER_ROW, CORNERS_PER_COL),
        corners,
        patternWasFound=True,
    )
    return img_cpy


def sharpness(arr):
    """Laplacian variance focus measure.

    Higher values mean sharper. See Pech-Pacheco et al., "Diatom
    autofocusing in brightfield microscopy: a comparative study",
    ICPR 2000.

    Args:
        arr: (H, W, C) uint8 array in [0, 255], with C = RGB.

    Returns:
        Variance of the Laplacian — higher means sharper.
    """
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def align_corner_ordering(pts_src, pts_dst):
    """Ensure src and dst corners use the same ordering.

    findChessboardCorners can start from opposite corners for different
    images of the same board, effectively flipping the entire ordering
    by 180°.  For an asymmetric board (rows != cols) there are only two
    possible orderings.  We detect the flip by comparing the first→last
    direction vectors and reverse dst if they disagree.
    """
    dir_src = pts_src[-1] - pts_src[0]
    dir_dst = pts_dst[-1] - pts_dst[0]
    if np.dot(dir_src, dir_dst) < 0:
        _logger.info("Corner ordering flipped — reversing dst points")
        pts_dst = pts_dst[::-1].copy()
    return pts_dst


def calc_homography(pts_src, pts_dst, ransac_reproj_threshold=3.0):
    """
    Args:
        ransac_reproj_threshold: RANSAC reprojection threshold, in the same
            units as pts_dst (e.g. mm if the points are in mm).
    """
    pts_dst = align_corner_ordering(pts_src, pts_dst)
    H, mask = cv2.findHomography(
        pts_src,
        pts_dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_reproj_threshold,
    )
    n_inliers = np.sum(mask)
    _logger.info(f"Homography inliers/outliers = {n_inliers}/{len(pts_src)}")
    return H


def decompose_homography(H, src_center=None, dst_center=None):
    """
    Decompose a 2D homography into intuitive geometric parameters.

    We treat H as mapping from stimulus plane to photo plane.
    The decomposition extracts:
      - rotation (in-plane)
      - scale / zoom (uniform + anisotropic)
      - tilt (perspective foreshortening, from pitch/yaw of camera)
      - translation (shift of pattern centre)

    Method: factor H into components via SVD and direct extraction.

    Args:
        src_center, dst_center: Optional (x, y) centres of the stimulus and
            photo planes, in the same units as the points H was fitted on.
            When given, the result includes center_dx/center_dy: the offset
            of the mapped stimulus centre from the photo centre, in photo
            units. This is invariant to where the plane origins sit and,
            unlike tx/ty, does not absorb rotation; it directly answers
            "how far is the pattern centre from the camera crosshair".
    """
    # Normalise so H[2,2] = 1
    H = H / H[2, 2]

    # ── Offset of the pattern centre from the photo centre ──
    center_dx = center_dy = None
    if src_center is not None and dst_center is not None:
        p = H @ np.array([src_center[0], src_center[1], 1.0])
        center_dx = p[0] / p[2] - dst_center[0]
        center_dy = p[1] / p[2] - dst_center[1]

    # ── Affine part: top-left 2×2 sub-block of H ──
    A = H[:2, :2]

    # SVD of the affine part: A = U @ diag(s) @ Vt
    U, s, Vt = np.linalg.svd(A)

    # ── Rotation ──
    # The rotation component is R = U @ Vt
    R = U @ Vt
    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        U[:, 1] *= -1
        R = U @ Vt
        s[1] *= -1

    rotation_rad = np.arctan2(R[1, 0], R[0, 0])
    rotation_deg = np.degrees(rotation_rad)

    # ── Scale / zoom ──
    # Singular values give the two principal scale factors.
    # Uniform zoom = geometric mean; anisotropy = ratio.
    scale_x, scale_y = s[0], s[1]
    zoom = np.sqrt(scale_x * scale_y)  # geometric mean
    aspect_ratio = scale_x / scale_y  # anisotropy (1.0 = isotropic)

    # ── Perspective / tilt ──
    # The perspective vector is h = [H[2,0], H[2,1]].
    # Its magnitude indicates how much perspective foreshortening
    # is present — i.e. how tilted the photo plane is relative to
    # the stimulus plane.
    #
    # For small tilts, |h| is proportional to tan(tilt) / focal_length.
    # Without knowing the focal length we report the raw perspective
    # coefficients and an "effective tilt" metric.
    h_persp = np.array([H[2, 0], H[2, 1]])
    perspective_magnitude = np.linalg.norm(h_persp)
    perspective_direction_deg = np.degrees(np.arctan2(h_persp[1], h_persp[0]))

    # ── Translation ──
    # Raw translation terms: where the stimulus origin (0, 0) maps in the
    # photo plane. Note that rotation and perspective leak into these, as
    # they describe a corner far from the pattern centre; center_dx/center_dy
    # are the intuitive alternative.
    tx, ty = H[0, 2], H[1, 2]

    return {
        "rotation_deg": rotation_deg,
        "zoom": zoom,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "aspect_ratio": aspect_ratio,
        "perspective_magnitude": perspective_magnitude,
        "perspective_direction_deg": perspective_direction_deg,
        "tx": tx,
        "ty": ty,
        "center_dx": center_dx,
        "center_dy": center_dy,
        "H": H,
    }
