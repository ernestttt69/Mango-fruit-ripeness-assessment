import cv2 as cv
import numpy as np
from skimage.feature import local_binary_pattern

# 1. IMAGE ENHANCEMENT: CLAGC
def apply_clagc(img_bgr):
    """
    Combines Gamma Correction and CLAHE
    for balanced brightness and local contrast.
    """

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)

    # Adaptive Gamma Adjustment
    mean_v = np.mean(v) / 255.0

    gamma = np.log(0.5) / np.log(mean_v + 1e-5)

    v_gamma = np.uint8(
        np.clip(
            np.power(v / 255.0, gamma) * 255.0,
            0,
            255
        )
    )

    # CLAHE
    clahe = cv.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    v_enhanced = clahe.apply(v_gamma)

    enhanced_hsv = cv.merge([
        h,
        s,
        v_enhanced
    ])

    enhanced_bgr = cv.cvtColor(
        enhanced_hsv,
        cv.COLOR_HSV2BGR
    )

    return enhanced_bgr

# 2. MULTI-SCALE RETINEX
def multi_scale_retinex(
    img,
    sigmas=[15, 80]
):
    """
    Shadow and illumination correction
    using Multi-Scale Retinex.
    """

    img_float = img.astype(
        np.float32
    ) + 1.0

    msr = np.zeros_like(
        img_float
    )

    for sigma in sigmas:

        blur = cv.GaussianBlur(
            img_float,
            (0, 0),
            sigma
        )

        msr += (
            np.log10(img_float)
            -
            np.log10(blur + 1e-5)
        )

    msr = msr / len(sigmas)

    # Normalize each colour channel
    for i in range(3):

        msr_c = msr[:, :, i]

        min_v = np.min(msr_c)
        max_v = np.max(msr_c)

        if max_v > min_v:

            msr[:, :, i] = (
                (msr_c - min_v)
                /
                (max_v - min_v)
                *
                255.0
            )

    return np.clip(
        msr,
        0,
        255
    ).astype(np.uint8)

# 3. DENOISING
def apply_nlm_denoising(img):
    """
    Fast bilateral denoising that
    preserves texture edges.
    """

    return cv.bilateralFilter(
        img,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

# 4. FOREGROUND SEGMENTATION
def apply_grabcut(img):
    """
    Foreground segmentation using
    Otsu thresholding and morphological cleanup.
    """

    gray = cv.cvtColor(
        img,
        cv.COLOR_BGR2GRAY
    )

    # Otsu automatic threshold
    _, mask = cv.threshold(
        gray,
        0,
        1,
        cv.THRESH_BINARY
        +
        cv.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv.getStructuringElement(
        cv.MORPH_ELLIPSE,
        (7, 7)
    )

    mask = cv.morphologyEx(
        mask,
        cv.MORPH_CLOSE,
        kernel
    )

    segmented_img = (
        img
        *
        mask[:, :, np.newaxis]
    )

    return segmented_img, mask

# 5. CIE LAB CONVERSION
def convert_to_cielab(img_bgr):
    """
    Convert image from BGR into
    CIE Lab colour space.
    """

    return cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2LAB
    )

# 6. LBP + COLOUR FEATURE EXTRACTION
def extract_high_accuracy_features(
    lab_img,
    mask
):
    """
    Extracts:

    1. LBP texture features
    2. Mean a* colour
    3. Standard deviation a*
    4. Mean b* colour
    5. Standard deviation b*
    """

    l_chan, a_chan, b_chan = cv.split(
        lab_img
    )

    # LBP Texture 
    radius = 2

    n_points = (
        8 * radius
    )

    lbp = local_binary_pattern(
        l_chan,
        n_points,
        radius,
        method="uniform"
    )

    n_bins_lbp = (
        n_points + 2
    )

    if np.any(mask > 0):

        lbp_masked = lbp[
            mask > 0
        ]

    else:

        lbp_masked = lbp.ravel()

    lbp_hist, _ = np.histogram(
        lbp_masked,
        bins=n_bins_lbp,
        range=(
            0,
            n_bins_lbp
        ),
        density=True
    )

    # CIE Lab Colour Statistics
    if np.any(mask > 0):

        valid_a = a_chan[
            mask > 0
        ]

        valid_b = b_chan[
            mask > 0
        ]

    else:

        valid_a = a_chan.ravel()

        valid_b = b_chan.ravel()

    color_stats = np.array([
        np.mean(valid_a),
        np.std(valid_a),
        np.mean(valid_b),
        np.std(valid_b)
    ])

    # Combine LBP + Lab features
    combined_features = np.hstack([
        lbp_hist,
        color_stats
    ])

    return combined_features, lbp

# 7. COMPLETE IMAGE PROCESSING PIPELINE
def process_mango_image(img_bgr):
    """
    Runs exactly the preprocessing sequence
    used by the SVM training pipeline.

    Input:
        OpenCV BGR image

    Output:
        features
        intermediate images
    """

    # Resize 
    crop = cv.resize(
        img_bgr,
        (224, 224)
    )

    # 1. Multi-Scale Retinex
    msr_img = multi_scale_retinex(
        crop
    )

    # 2. CLAGC
    enhanced_img = apply_clagc(
        msr_img
    )

    # 3. Denoising
    denoised_img = apply_nlm_denoising(
        enhanced_img
    )

    # 4. Segmentation
    segmented_img, mask = apply_grabcut(
        denoised_img
    )

    # 5. CIE Lab
    lab_img = convert_to_cielab(
        segmented_img
    )

    # 6. Feature extraction
    features, lbp_viz = (
        extract_high_accuracy_features(
            lab_img,
            mask
        )
    )

    return {
        "features": features,
        "original": crop,
        "msr": msr_img,
        "clagc": enhanced_img,
        "denoised": denoised_img,
        "segmented": segmented_img,
        "lab": lab_img,
        "lbp": lbp_viz,
        "mask": mask
    }