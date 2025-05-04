from typing import Dict, Tuple

import cv2
import numpy as np


def analyse_region_colour(
    image: np.ndarray, mask: np.ndarray, bins: Tuple[int, int, int] = (180, 256, 256)
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute the H, S, V histograms of the masked region in an RGB image.

    Parameters:
        image (np.ndarray): Input image in RGB format, shape (H, W, 3), dtype=uint8.
        mask (np.ndarray): Binary mask, shape (H, W), dtype=uint8 or bool.
                           Non-zero (or True) pixels define the region of interest.
        bins (tuple): Number of bins for H, S and V channels respectively.
                      Defaults to (180, 256, 256).

    Returns:
        Dict[str, (hist, bin_edges)]:
            'H': (hist_H, edges_H)
            'S': (hist_S, edges_S)
            'V': (hist_V, edges_V)
        Each `hist_*` is a 1D array of counts, and `edges_*` the corresponding bin edges.
    """
    # Ensure mask is boolean
    mask_bool = mask.astype(bool)

    # Convert RGB→HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Extract per-channel values under the mask
    h_vals = hsv[:, :, 0][mask_bool].ravel()
    s_vals = hsv[:, :, 1][mask_bool].ravel()
    v_vals = hsv[:, :, 2][mask_bool].ravel()

    if h_vals.size == 0:
        raise ValueError("Mask selects no pixels")

    # Compute histograms
    hist_h, edges_h = np.histogram(h_vals, bins=bins[0], range=(0, 180))
    hist_s, edges_s = np.histogram(s_vals, bins=bins[1], range=(0, 256))
    hist_v, edges_v = np.histogram(v_vals, bins=bins[2], range=(0, 256))

    return {"H": (hist_h, edges_h), "S": (hist_s, edges_s), "V": (hist_v, edges_v)}


def apply_hsv_channel_offset(
    image_bgr: np.ndarray, mask: np.ndarray, dh: float = 0, ds: float = 0, dv: float = 0
) -> np.ndarray:
    """
    Add constant offsets to H, S, V channels inside a mask.

    Parameters:
        image_bgr (np.ndarray): Input image in BGR (uint8).
        mask (np.ndarray): Binary mask (0 or 255), shape (H, W).
        dh (float): Offset to add to Hue (wrapped mod 180).
        ds (float): Offset to add to Saturation (clamped 0–255).
        dv (float): Offset to add to Value (clamped 0–255).

    Returns:
        np.ndarray: Resulting BGR image (uint8).
    """
    # Convert to HSV float32
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 2D boolean mask
    mask_bool = mask > 0

    # Channel-wise apply offsets under mask
    # Hue channel (0): wrap modulo 180
    h = hsv[:, :, 0]
    h[mask_bool] = (h[mask_bool] + dh) % 180

    # Saturation channel (1): clamp 0–255
    s = hsv[:, :, 1]
    s[mask_bool] = np.clip(s[mask_bool] + ds, 0, 255)

    # Value channel (2): clamp 0–255
    v = hsv[:, :, 2]
    v[mask_bool] = np.clip(v[mask_bool] + dv, 0, 255)

    # Write back and convert
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v

    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def colourise_image(
    image: np.ndarray, hue: int = 180, saturation: int = 200, lightness: int = 128
):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    # Assign hue, saturation, and value
    hsv[..., 0] = hue  # Hue: 0-179 in OpenCV
    hsv[..., 1] = saturation  # Saturation: 0-255
    hsv[..., 2] = gray  # Value from grayscale image

    # Convert back to BGR for viewing
    colourised = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return colourised
