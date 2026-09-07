import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from skimage.feature import local_binary_pattern
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


# =========================================================
# 1. PATH SETUP
# =========================================================

DATASET_PATH = Path(r"Z:\mangoDataSet")
SPLITS = ["train", "valid", "test"]


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
        clipLimit=1.0,
        tileGridSize=(4, 4)
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
        d=3,
        sigmaColor=20,
        sigmaSpace=20
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
            3,
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
        (3, 3)
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
        [8],
        [0, 256]
    ).flatten()

    b_hist = cv.calcHist(
        [B],
        [0],
        mask_cv,
        [8],
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
        [9],
        [0, 180]
    ).flatten()

    s_hist = cv.calcHist(
        [S],
        [0],
        mask_cv,
        [4],
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
    """
    Approximate colour ratios in HSV space.
    Thresholds may later be tuned using your dataset.
    """

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

    # OpenCV Hue range = 0-179

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

    # Dark/brown-ish / overripe indicator
    dark = (
        v < 90
    )

    total = len(h)

    green_ratio = np.sum(green) / total
    yellow_ratio = np.sum(yellow) / total
    dark_ratio = np.sum(dark) / total

    return np.array([
        green_ratio,
        yellow_ratio,
        dark_ratio
    ], dtype=np.float32)


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

    radius = 1
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


# =========================================================
# 11. DATASET ARRAYS
# =========================================================

X_train, y_train = [], []
X_valid, y_valid = [], []
X_test, y_test = [], []


# =========================================================
# 12. READ DATASET
# =========================================================

display_count = 0

for split in SPLITS:

    folder = DATASET_PATH / split

    if not folder.exists():
        print(
            f"Skipping missing folder: {folder}"
        )
        continue

    image_files = list(
        folder.glob("*.jpg")
    )

    print(
        f"\nFound {len(image_files)} images "
        f"in '{split}' set."
    )

    for img_path in tqdm(
        image_files,
        desc=f"Processing {split}"
    ):

        img = cv.imread(
            str(img_path)
        )

        if img is None:
            continue

        xml_path = img_path.with_suffix(
            ".xml"
        )

        if not xml_path.exists():
            continue

        try:
            tree = ET.parse(
                xml_path
            )

            root = tree.getroot()

        except ET.ParseError:
            continue

        img_h, img_w = img.shape[:2]

        for obj in root.findall(
            "object"
        ):

            name_node = obj.find(
                "name"
            )

            b = obj.find(
                "bndbox"
            )

            if name_node is None or b is None:
                continue

            label = name_node.text.strip()

            try:
                xmin = int(float(
                    b.find("xmin").text
                ))

                ymin = int(float(
                    b.find("ymin").text
                ))

                xmax = int(float(
                    b.find("xmax").text
                ))

                ymax = int(float(
                    b.find("ymax").text
                ))

            except Exception:
                continue

            # Keep box inside image
            xmin = max(
                0,
                min(xmin, img_w - 1)
            )

            ymin = max(
                0,
                min(ymin, img_h - 1)
            )

            xmax = max(
                1,
                min(xmax, img_w)
            )

            ymax = max(
                1,
                min(ymax, img_h)
            )

            if xmax <= xmin or ymax <= ymin:
                continue

            crop = img[
                ymin:ymax,
                xmin:xmax
            ]

            if crop.size == 0:
                continue


            # =================================================
            # OPTIONAL QUALITY FILTER
            # =================================================

            crop_h, crop_w = crop.shape[:2]

            # Ignore extremely tiny mango ROI
            if crop_w < 10 or crop_h < 20:
                continue


            # =================================================
            # FEATURE EXTRACTION
            # =================================================

            (
                features,
                resized,
                corrected,
                denoised,
                mask,
                segmented,
                lbp_img

            ) = extract_features(
                crop
            )


            # =================================================
            # APPEND DATA
            # =================================================

            if split == "train":

                X_train.append(
                    features
                )

                y_train.append(
                    label
                )

            elif split == "valid":

                X_valid.append(
                    features
                )

                y_valid.append(
                    label
                )

            elif split == "test":

                X_test.append(
                    features
                )

                y_test.append(
                    label
                )


            # =================================================
            # DISPLAY FIRST 5 EXAMPLES
            # =================================================

            if display_count < 5:

                fig, axes = plt.subplots(
                    1,
                    7,
                    figsize=(22, 4)
                )

                axes[0].imshow(
                    cv.cvtColor(
                        crop,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[0].set_title(
                    "1. Original ROI"
                )

                axes[1].imshow(
                    cv.cvtColor(
                        resized,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[1].set_title(
                    "2. Resize + Padding"
                )

                axes[2].imshow(
                    cv.cvtColor(
                        corrected,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[2].set_title(
                    "3. Illumination"
                )

                axes[3].imshow(
                    cv.cvtColor(
                        denoised,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[3].set_title(
                    "4. Denoised"
                )

                axes[4].imshow(
                    mask,
                    cmap="gray"
                )

                axes[4].set_title(
                    "5. Mask"
                )

                axes[5].imshow(
                    cv.cvtColor(
                        segmented,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[5].set_title(
                    "6. Segmented"
                )

                axes[6].imshow(
                    lbp_img,
                    cmap="gray"
                )

                axes[6].set_title(
                    "7. LBP"
                )

                for ax in axes:
                    ax.axis(
                        "off"
                    )

                fig.suptitle(
                    f"{split} | "
                    f"{img_path.name} | "
                    f"{label}"
                )

                plt.tight_layout()
                plt.show()

                display_count += 1


# =========================================================
# 13. CONVERT TO NUMPY
# =========================================================

X_train = np.array(
    X_train,
    dtype=np.float32
)

y_train = np.array(
    y_train
)

X_valid = np.array(
    X_valid,
    dtype=np.float32
)

y_valid = np.array(
    y_valid
)

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test
)


# =========================================================
# 14. DATASET SUMMARY
# =========================================================

print(
    "\n========== FINAL DATASET =========="
)

print(
    "Training:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_valid.shape,
    y_valid.shape
)

print(
    "Testing:",
    X_test.shape,
    y_test.shape
)


print(
    "\n========== CLASS DISTRIBUTION =========="
)

print(
    "\nTrain:",
    Counter(y_train)
)

print(
    "Validation:",
    Counter(y_valid)
)

print(
    "Test:",
    Counter(y_test)
)


# =========================================================
# 15. FEATURE SIZE
# =========================================================

if len(X_train) > 0:

    print(
        "\nFeatures per mango:",
        X_train.shape[1]
    )

# %%

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

train_pred = svm_model.predict(X_train_scaled)
valid_pred = svm_model.predict(X_valid_scaled)
test_pred = svm_model.predict(X_test_scaled)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Validation Accuracy:",accuracy_score(y_valid, valid_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))

# %%

print(classification_report(y_test, test_pred))

# %%

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

plt.figure(figsize=(5,5))

plt.bar(
    ["Training", "Testing"],
    [train_acc, test_acc]
)

plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("SVM Classification Accuracy")

plt.show()

# %%

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, test_pred)
labels = ["Unripe", "Ripe", "Over Ripe"]

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Final Results Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# %%

accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average="weighted")
recall = recall_score(y_test, test_pred, average="weighted")
f1 = f1_score(y_test, test_pred, average="weighted")

print("----- Final Summary Metrics -----")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")

# %%

joblib.dump(svm_model, "mango_svm_model.pkl")
joblib.dump(scaler, "mango_scaler.pkl")

print("\nModel and scaler saved successfully!")
