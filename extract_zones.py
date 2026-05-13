"""
Programmatic retinal zone extraction — self-contained module.

Mirrors Slice Zone Mohammad 5.1.2026 logic (auto fovea/crosshair, yellow removal,
10 zones), but returns cropped RGBA arrays only — no PNG folder output.

Requirements:
    pip install pillow numpy opencv-python

Public API:
    extract(image_path) -> list[np.ndarray]
"""

from __future__ import annotations

import math
import os
from typing import List

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
except Exception as exc:
    raise ImportError(
        "OpenCV (cv2) failed to import.\n\n"
        "Install it with:\n"
        "  pip install opencv-python\n"
    ) from exc

# ---------------------------------------------------------------------------
# Defaults (same as original slicer CLI defaults)
# ---------------------------------------------------------------------------
DEFAULT_CX = None       # auto from yellow crosshair
DEFAULT_CY = None
PX_PER_MM = 53
INNER_R_MM = 3.0
OUTER_R_MM = 16.0
ANGLE_DEG = None      # auto from yellow crosshair
ONH_OFFSET_X = 270
ONH_OFFSET_Y = 0
ONH_RX = 80
ONH_RY = 95

__all__ = ["extract"]


def resolve_path(image_path):
    """Resolve relative image paths against cwd and this file's directory."""
    if os.path.isabs(image_path):
        return image_path
    if os.path.exists(image_path):
        return image_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, image_path)
    if os.path.exists(candidate):
        return candidate
    return image_path


def make_yellow_mask(arr):
    rgb = arr[:, :, :3].astype(np.int16)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    yellow = (
        (r > 40)
        & (g > 40)
        & (b < 100)
        & (np.abs(r - g) < 85)
        & ((r - b) > 20)
        & ((g - b) > 20)
    )
    return yellow


def remove_yellow_overlay(arr, inpaint_radius=3, dilate_iterations=1):
    yellow = make_yellow_mask(arr)
    if int(yellow.sum()) == 0:
        return arr.copy()
    mask = (yellow.astype(np.uint8) * 255)
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3].copy()
    cleaned_rgb = cv2.inpaint(rgb, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    cleaned = arr.copy()
    cleaned[:, :, :3] = cleaned_rgb
    cleaned[:, :, 3] = alpha
    return cleaned


def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return float(px), float(py)


def normalize_angle_deg(angle):
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def detect_crosshair_from_yellow(arr, output_dir=None, save_debug=False):
    H, W = arr.shape[:2]
    yellow = make_yellow_mask(arr)
    yellow_count = int(yellow.sum())
    if yellow_count < 200:
        raise ValueError(
            "Could not find enough yellow overlay pixels. Make sure the image "
            "contains yellow circles/crosshair."
        )
    mask_u8 = (yellow.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    min_len = max(80, int(min(H, W) * 0.08))
    lines = cv2.HoughLinesP(
        mask_u8,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=35,
    )
    if lines is None or len(lines) < 2:
        raise ValueError("Yellow overlay found, but crosshair lines could not be detected.")
    horizontal = []
    vertical = []
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, item)
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length < min_len:
            continue
        angle = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        if abs(angle) <= 25:
            horizontal.append((length, angle, (x1, y1, x2, y2)))
        elif abs(abs(angle) - 90) <= 25:
            vertical.append((length, angle, (x1, y1, x2, y2)))
    if not horizontal or not vertical:
        raise ValueError("Could not separate horizontal and vertical yellow crosshair lines.")
    _, h_angle, h_line = max(horizontal, key=lambda x: x[0])
    _, _, v_line = max(vertical, key=lambda x: x[0])
    point = line_intersection(h_line, v_line)
    if point is None:
        raise ValueError("Detected crosshair lines are parallel; cannot estimate fovea.")
    cx, cy = point
    if not (0 <= cx < W and 0 <= cy < H):
        raise ValueError(f"Detected fovea is outside image bounds: ({cx:.1f}, {cy:.1f})")
    angle_deg = h_angle
    if save_debug and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        debug_img = Image.fromarray(arr).convert("RGBA")
        draw = ImageDraw.Draw(debug_img)
        draw.line(h_line, fill=(255, 0, 0, 255), width=4)
        draw.line(v_line, fill=(0, 255, 0, 255), width=4)
        r = 18
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(255, 0, 0, 255), width=5)
        draw.text(
            (cx + 25, cy + 25),
            f"fovea=({cx:.0f},{cy:.0f}), angle={angle_deg:.2f} deg",
            fill=(255, 0, 0, 255),
        )
        debug_img.save(os.path.join(output_dir, "debug_detected_fovea.png"))
    return int(round(cx)), int(round(cy)), float(angle_deg), yellow_count


def make_masks(width, height, cx, cy, r_inner, r_outer,
               angle_deg, onh_offset_x, onh_offset_y, onh_rx, onh_ry):
    yy, xx = np.ogrid[:height, :width]
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    upper = ry <= 0
    lower = ry > 0
    nasal = rx <= 0
    temporal = rx > 0
    in_inner = dist <= r_inner
    in_ring = (dist > r_inner) & (dist <= r_outer)
    outside = dist > r_outer
    onh_cx = cx + onh_offset_x
    onh_cy = cy + onh_offset_y
    dx_onh = xx - onh_cx
    dy_onh = yy - onh_cy
    in_onh = (dx_onh**2 / onh_rx**2 + dy_onh**2 / onh_ry**2) <= 1.0
    return {
        "Zone_01_inner_upper_nasal": in_inner & upper & nasal,
        "Zone_02_inner_upper_temporal": in_inner & upper & temporal,
        "Zone_03_inner_lower_temporal": in_inner & lower & temporal,
        "Zone_04_inner_lower_nasal": in_inner & lower & nasal,
        "Zone_05_ring_upper_nasal": in_ring & upper & nasal,
        "Zone_06_ring_upper_temporal": in_ring & upper & temporal,
        "Zone_07_ring_lower_temporal": in_ring & lower & temporal,
        "Zone_08_ring_lower_nasal": in_ring & lower & nasal,
        "Zone_09_optic_disc": in_onh,
        "Zone_10_far_periphery": outside,
    }


def crop_to_content(img_array):
    alpha = img_array[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)
    if not rows.any() or not cols.any():
        return img_array
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img_array[rmin : rmax + 1, cmin : cmax + 1]


def extract(image_path: str) -> List[np.ndarray]:
    """
    Load an image, detect fovea/crosshair (when defaults are None), remove yellow
    overlay, slice into 10 zones, crop each to content.

    Returns a list of 10 RGBA uint8 arrays, Zone_01 … Zone_10 order.
    """
    path = resolve_path(image_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input image not found: {path}")

    img = Image.open(path).convert("RGBA")
    original_arr = np.array(img)
    h, w = original_arr.shape[:2]

    cx, cy, angle_deg = DEFAULT_CX, DEFAULT_CY, ANGLE_DEG
    if cx is None or cy is None or angle_deg is None:
        detected_cx, detected_cy, detected_angle, _ = detect_crosshair_from_yellow(
            original_arr,
            output_dir=None,
            save_debug=False,
        )
        if cx is None:
            cx = detected_cx
        if cy is None:
            cy = detected_cy
        if angle_deg is None:
            angle_deg = detected_angle

    arr_for_output = remove_yellow_overlay(original_arr, inpaint_radius=3, dilate_iterations=1)
    r_inner = INNER_R_MM * PX_PER_MM
    r_outer = OUTER_R_MM * PX_PER_MM
    masks = make_masks(
        w, h, cx, cy, r_inner, r_outer,
        angle_deg, ONH_OFFSET_X, ONH_OFFSET_Y, ONH_RX, ONH_RY,
    )

    out: List[np.ndarray] = []
    for zone_name in sorted(masks.keys()):
        mask = masks[zone_name]
        zone_arr = arr_for_output.copy()
        zone_arr[~mask, 3] = 0
        out.append(crop_to_content(zone_arr))
    return out
