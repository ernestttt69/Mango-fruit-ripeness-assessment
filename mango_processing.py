import cv2 as cv
import numpy as np

from skimage.feature import local_binary_pattern
from tqdm.notebook import tqdm


# =========================================================
# 2. RESIZE WITH ASPECT RATIO + PADDING
# =========================================================

def resize_with_padding(img, target_size=224):
    """
    Resize image while preserving aspect ratio.
    Padding is added instead of stretching the mango.
    """

    h, w = img.shape[:2]

    if h == 0 or w == 0:
        return None

    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    interpolation = (
        cv.INTER_AREA
        if scale < 1
        else cv.INTER_CUBIC
    )

    resized = cv.resize(
        img,
        (new_w, new_h),
        interpolation=interpolation
    )

    canvas = np.zeros(
        (target_size, target_size, 3),
        dtype=np.uint8
    )

    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# =========================================================
# 3. ILLUMINATION CORRECTION
# =========================================================

def illumination_correction(img_bgr):
    """
    Mild illumination correction.
    Avoid aggressive Retinex because ripeness depends on colour.
    """

    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    L, A, B = cv.split(lab)

    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    L_enhanced = clahe.apply(L)

    corrected_lab = cv.merge(
        [L_enhanced, A, B]
    )

    corrected = cv.cvtColor(
        corrected_lab,
        cv.COLOR_LAB2BGR
    )

    return corrected


# =========================================================
# 4. DENOISING
# =========================================================

def denoise_image(img_bgr):
    """
    Mild bilateral filtering.
    Preserves edges and colour better than aggressive smoothing.
    """

    return cv.bilateralFilter(
        img_bgr,
        d=5,
        sigmaColor=40,
        sigmaSpace=40
    )


# =========================================================
# 5. FOREGROUND MASK
# =========================================================

def create_mango_mask(img_bgr):
    """
    GrabCut foreground segmentation.
    Since XML already provides mango ROI,
    mango should be near the centre.
    """

    h, w = img_bgr.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    margin_x = max(2, int(w * 0.03))
    margin_y = max(2, int(h * 0.03))

    rect_w = w - 2 * margin_x
    rect_h = h - 2 * margin_y

    if rect_w <= 0 or rect_h <= 0:
        return np.ones((h, w), dtype=np.uint8)

    rect = (
        margin_x,
        margin_y,
        rect_w,
        rect_h
    )

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:
        cv.grabCut(
            img_bgr,
            mask,
            rect,
            bg_model,
            fg_model,
            5,
            cv.GC_INIT_WITH_RECT
        )

        binary_mask = np.where(
            (mask == cv.GC_FGD) |
            (mask == cv.GC_PR_FGD),
            1,
            0
        ).astype(np.uint8)

    except cv.error:
        binary_mask = np.ones(
            (h, w),
            dtype=np.uint8
        )

    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (5, 5)
    )

    binary_mask = cv.morphologyEx(
        binary_mask,
        cv.MORPH_CLOSE,
        kernel
    )

    binary_mask = cv.morphologyEx(
        binary_mask,
        cv.MORPH_OPEN,
        kernel
    )

    if np.sum(binary_mask) < 50:
        binary_mask = np.ones(
            (h, w),
            dtype=np.uint8
        )

    return binary_mask


# =========================================================
# 6. LAB COLOR FEATURES
# =========================================================

def extract_lab_features(img_bgr, mask):
    """
    Extract Lab colour statistics + histograms.
    a* represents green-red information.
    b* represents blue-yellow information.
    """

    lab = cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2LAB
    )

    L, A, B = cv.split(lab)

    valid = mask > 0

    if not np.any(valid):
        valid = np.ones_like(mask, dtype=bool)

    a_values = A[valid]
    b_values = B[valid]

    # Statistics
    lab_stats = np.array([
        np.mean(a_values),
        np.std(a_values),
        np.mean(b_values),
        np.std(b_values)
    ], dtype=np.float32)

    mask_cv = (
        mask.astype(np.uint8) * 255
    )

    # Histograms
    a_hist = cv.calcHist(
        [A],
        [0],
        mask_cv,
        [16],
        [0, 256]
    ).flatten()

    b_hist = cv.calcHist(
        [B],
        [0],
        mask_cv,
        [16],
        [0, 256]
    ).flatten()

    if np.sum(a_hist) > 0:
        a_hist = a_hist / np.sum(a_hist)

    if np.sum(b_hist) > 0:
        b_hist = b_hist / np.sum(b_hist)

    return np.hstack([
        lab_stats,
        a_hist,
        b_hist
    ])


# =========================================================
# 7. HSV COLOR FEATURES
# =========================================================

def extract_hsv_features(img_bgr, mask):
    """
    HSV histogram useful for distinguishing
    green / yellow / brown colour distributions.
    """

    hsv = cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2HSV
    )

    H, S, V = cv.split(hsv)

    mask_cv = (
        mask.astype(np.uint8) * 255
    )

    h_hist = cv.calcHist(
        [H],
        [0],
        mask_cv,
        [18],
        [0, 180]
    ).flatten()

    s_hist = cv.calcHist(
        [S],
        [0],
        mask_cv,
        [8],
        [0, 256]
    ).flatten()

    if np.sum(h_hist) > 0:
        h_hist = h_hist / np.sum(h_hist)

    if np.sum(s_hist) > 0:
        s_hist = s_hist / np.sum(s_hist)

    return np.hstack([
        h_hist,
        s_hist
    ])


# =========================================================
# 8. GREEN / YELLOW / DARK PIXEL RATIOS
# =========================================================

def extract_color_ratios(img_bgr, mask):

    hsv = cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2HSV
    )

    H, S, V = cv.split(hsv)

    valid = mask > 0

    if not np.any(valid):
        return np.zeros(
            3,
            dtype=np.float32
        )

    h = H[valid]
    s = S[valid]
    v = V[valid]

    # Detect green mango pixels
    green = (
        (h >= 35) &
        (h <= 85) &
        (s > 40) &
        (v > 40)
    )

    # Detect yellow and orange ripe mango pixels
    yellow = (
        (h >= 10) &
        (h < 35) &
        (s > 40) &
        (v > 60)
    )

    # Detect dark or brown overripe pixels
    dark = (
        v < 90
    )

    total = len(h)

    green_ratio = (
        np.sum(green) / total
    )

    yellow_ratio = (
        np.sum(yellow) / total
    )

    dark_ratio = (
        np.sum(dark) / total
    )

    return np.array(
        [
            green_ratio,
            yellow_ratio,
            dark_ratio
        ],
        dtype=np.float32
    )


# =========================================================
# 9. LBP TEXTURE FEATURES
# =========================================================

def extract_lbp_features(img_bgr, mask):
    """
    LBP texture feature.
    Used as supplementary feature only.
    """

    gray = cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2GRAY
    )

    radius = 2
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform"
    )

    valid = mask > 0

    if np.any(valid):
        lbp_values = lbp[valid]
    else:
        lbp_values = lbp.ravel()

    n_bins = n_points + 2

    hist, _ = np.histogram(
        lbp_values,
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    return hist.astype(np.float32), lbp


# =========================================================
# 10. COMBINED FEATURE EXTRACTION
# =========================================================

def extract_features(original_crop):
    """
    Main feature extraction pipeline.

    Colour features are extracted from mildly corrected image.
    Texture features are supplementary.
    """

    resized = resize_with_padding(
        original_crop,
        target_size=224
    )

    corrected = illumination_correction(
        resized
    )

    denoised = denoise_image(
        corrected
    )

    mask = create_mango_mask(
        denoised
    )

    # Colour
    lab_features = extract_lab_features(
        corrected,
        mask
    )

    hsv_features = extract_hsv_features(
        corrected,
        mask
    )

    color_ratios = extract_color_ratios(
        corrected,
        mask
    )

    # Texture
    lbp_features, lbp_img = extract_lbp_features(
        denoised,
        mask
    )

    combined = np.hstack([
        lab_features,
        hsv_features,
        color_ratios,
        lbp_features
    ])

    segmented = (
        denoised *
        mask[:, :, np.newaxis]
    )

    return (
        combined,
        resized,
        corrected,
        denoised,
        mask,
        segmented,
        lbp_img
    )
    
def process_mango_image(image_bgr):
    if image_bgr is None:
        raise ValueError(
            "Image could not be decoded."
        )

    image_bgr = np.asarray(image_bgr)

    if image_bgr.ndim != 3:
        raise ValueError(
            f"Expected a 3D BGR image, "
            f"received shape {image_bgr.shape}"
        )

    if image_bgr.shape[2] != 3:
        raise ValueError(
            f"Expected 3 colour channels, "
            f"received shape {image_bgr.shape}"
        )

    (
        features,
        resized,
        corrected,
        denoised,
        mask,
        segmented,
        lbp_img
    ) = extract_features(
        image_bgr
    )

    features = np.asarray(
        features,
        dtype=np.float32
    ).flatten()

    if features.shape[0] != 83:
        raise ValueError(
            f"Expected 83 features, "
            f"but got {features.shape[0]}."
        )

    # CIE Lab image for app visualization
    lab_img = cv.cvtColor(
        corrected,
        cv.COLOR_BGR2LAB
    )

    return {
        "features": features,
        "original": resized,
        "corrected": corrected,
        "denoised": denoised,
        "mask": mask,
        "segmented": segmented,
        "lab": lab_img,
        "lbp": lbp_img,
    }