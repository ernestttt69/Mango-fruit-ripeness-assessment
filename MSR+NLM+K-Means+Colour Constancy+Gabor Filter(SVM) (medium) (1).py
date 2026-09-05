import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

import cv2 as cv
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm import tqdm
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
# 3. MULTI-SCALE RETINEX (MSR)
# =========================================================

def illumination_correction(img_bgr):
    """
    Multi-Scale Retinex (MSR).

    Reduces uneven illumination using multiple
    Gaussian scales.
    """

    img = img_bgr.astype(np.float32) + 1.0

    sigma_list = [15, 80, 250]

    msr = np.zeros_like(
        img,
        dtype=np.float32
    )

    for sigma in sigma_list:

        blurred = cv.GaussianBlur(
            img,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma
        )

        retinex = (
            np.log(img)
            - np.log(blurred + 1.0)
        )

        msr += retinex

    msr /= len(sigma_list)

    result = np.zeros_like(
        msr,
        dtype=np.float32
    )

    # Normalize each BGR channel independently
    for c in range(3):

        channel = msr[:, :, c]

        channel = cv.normalize(
            channel,
            None,
            0,
            255,
            cv.NORM_MINMAX
        )

        result[:, :, c] = channel

    result = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)

    # Reduce the MSR strength to avoid an over-bright result.
    result = cv.addWeighted(
        result,
        0.2,
        img_bgr,
        0.8,
        0
    )

    return result


# =========================================================
# 4. NON-LOCAL MEANS (NLM) DENOISING
# =========================================================

def denoise_image(img_bgr):
    """
    Non-Local Means (NLM) denoising.

    Reduces image noise while preserving
    useful surface details.
    """

    return cv.fastNlMeansDenoisingColored(
        img_bgr,
        None,
        h=1,
        hColor=1,
        templateWindowSize=7,
        searchWindowSize=21
    )


# =========================================================
# 5. K-MEANS SEGMENTATION
# =========================================================

def create_mango_mask(img_bgr):
    """
    K-Means colour clustering segmentation.

    Since XML already provides the mango ROI,
    the mango is expected to occupy the centre
    region of the cropped image.
    """

    h, w = img_bgr.shape[:2]

    pixels = img_bgr.reshape(
        (-1, 3)
    ).astype(np.float32)

    criteria = (
        cv.TERM_CRITERIA_EPS
        + cv.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )

    K = 3

    _, labels, centers = cv.kmeans(
        pixels,
        K,
        None,
        criteria,
        3,
        cv.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(
        (h, w)
    )

    # =====================================================
    # Select cluster that dominates centre of ROI
    # =====================================================

    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    x1 = int(w * 0.25)
    x2 = int(w * 0.75)

    centre_labels = labels[
        y1:y2,
        x1:x2
    ].flatten()

    counts = np.bincount(
        centre_labels,
        minlength=K
    )

    mango_cluster = np.argmax(
        counts
    )

    binary_mask = (
        labels == mango_cluster
    ).astype(np.uint8)

    # =====================================================
    # MORPHOLOGICAL CLEANUP
    # =====================================================

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

    # Same safety fallback concept
    if np.sum(binary_mask) < 50:

        binary_mask = np.ones(
            (h, w),
            dtype=np.uint8
        )

    return binary_mask


# =========================================================
# 6. COLOUR CONSTANCY
# =========================================================

def apply_colour_constancy(img_bgr):
    """
    Gray World Colour Constancy.

    Attempts to reduce colour cast caused
    by different illumination conditions.
    """

    img = img_bgr.astype(
        np.float32
    )

    b_mean = np.mean(
        img[:, :, 0]
    )

    g_mean = np.mean(
        img[:, :, 1]
    )

    r_mean = np.mean(
        img[:, :, 2]
    )

    gray_mean = (
        b_mean
        + g_mean
        + r_mean
    ) / 3.0

    eps = 1e-6

    img[:, :, 0] *= (
        gray_mean /
        (b_mean + eps)
    )

    img[:, :, 1] *= (
        gray_mean /
        (g_mean + eps)
    )

    img[:, :, 2] *= (
        gray_mean /
        (r_mean + eps)
    )

    corrected = np.clip(
        img,
        0,
        255
    ).astype(np.uint8)

    corrected = cv.addWeighted(
        corrected,
        0.25,
        img_bgr,
        0.75,
        0
    )

    return corrected


def extract_colour_constancy_features(
    img_bgr,
    mask
):
    """
    Extract colour features after
    Gray World Colour Constancy.

    Mean and standard deviation of
    each colour channel are used.
    """

    colour_corrected = (
        apply_colour_constancy(
            img_bgr
        )
    )

    valid = mask > 0

    if not np.any(valid):

        valid = np.ones_like(
            mask,
            dtype=bool
        )

    features = []

    for c in range(3):

        values = (
            colour_corrected[
                :, :, c
            ][valid]
        )

        features.extend([
            np.mean(values),
            np.std(values)
        ])

    features = np.array(
        features,
        dtype=np.float32
    )

    return (
        features,
        colour_corrected
    )


# =========================================================
# 7. GABOR FILTER TEXTURE FEATURES
# =========================================================

def extract_gabor_features(
    img_bgr,
    mask
):
    """
    Gabor Filter texture feature extraction.

    Multiple orientations are used to detect
    mango surface texture, spots and wrinkles.
    """

    gray = cv.cvtColor(
        img_bgr,
        cv.COLOR_BGR2GRAY
    )

    gray = gray.astype(
        np.float32
    )

    valid = mask > 0

    if not np.any(valid):

        valid = np.ones_like(
            mask,
            dtype=bool
        )

    features = []

    gabor_responses = []

    orientations = [
        0,
        np.pi / 4,
        np.pi / 2,
        3 * np.pi / 4
    ]

    for theta in orientations:

        kernel = cv.getGaborKernel(
            ksize=(15, 15),
            sigma=3.0,
            theta=theta,
            lambd=8.0,
            gamma=0.5,
            psi=0,
            ktype=cv.CV_32F
        )

        response = cv.filter2D(
            gray,
            cv.CV_32F,
            kernel
        )

        values = response[
            valid
        ]

        features.extend([
            np.mean(values),
            np.std(values)
        ])

        gabor_responses.append(
            np.abs(response)
        )

    features = np.array(
        features,
        dtype=np.float32
    )

    # Visualization image
    gabor_img = np.mean(
        gabor_responses,
        axis=0
    )

    gabor_img = cv.normalize(
        gabor_img,
        None,
        0,
        255,
        cv.NORM_MINMAX
    ).astype(np.uint8)

    return (
        features,
        gabor_img
    )


# =========================================================
# 8. COMBINED FEATURE EXTRACTION
# =========================================================

def extract_features(original_crop):
    """
    Siyu processing pipeline.

    1. Resize + Padding
    2. Multi-Scale Retinex
    3. Non-Local Means
    4. K-Means Segmentation
    5. Colour Constancy
    6. Gabor Filter
    """

    # =====================================================
    # RESIZE
    # =====================================================

    resized = resize_with_padding(
        original_crop,
        target_size=224
    )

    # =====================================================
    # MULTI-SCALE RETINEX
    # =====================================================

    corrected = illumination_correction(
        resized
    )

    # =====================================================
    # NON-LOCAL MEANS
    # =====================================================

    denoised = denoise_image(
        corrected
    )

    # =====================================================
    # K-MEANS SEGMENTATION
    # =====================================================

    mask = create_mango_mask(
        denoised
    )

    # =====================================================
    # COLOUR CONSTANCY
    # =====================================================

    (
        colour_features,
        colour_corrected

    ) = extract_colour_constancy_features(
        denoised,
        mask
    )

    # =====================================================
    # GABOR TEXTURE
    # =====================================================

    (
        gabor_features,
        gabor_img

    ) = extract_gabor_features(
        colour_corrected,
        mask
    )

    # =====================================================
    # COMBINE FEATURES
    # =====================================================

    combined = np.hstack([
        colour_features,
        gabor_features
    ])

    return (
        combined,
        resized,
        corrected,
        denoised,
        mask,
        colour_corrected,
        gabor_img
    )


# =========================================================
# 9. DATASET ARRAYS
# =========================================================

X_train, y_train = [], []
X_valid, y_valid = [], []
X_test, y_test = [], []


# =========================================================
# 10. READ DATASET
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

            if (
                name_node is None
                or b is None
            ):
                continue

            label = (
                name_node.text.strip()
            )

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

            # =================================================
            # KEEP BOX INSIDE IMAGE
            # =================================================

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

            if (
                xmax <= xmin
                or ymax <= ymin
            ):
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
                colour_corrected,
                gabor_img

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

                # ---------------------------------------------
                # 1. ORIGINAL ROI
                # ---------------------------------------------

                axes[0].imshow(
                    cv.cvtColor(
                        crop,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[0].set_title(
                    "1. Original ROI"
                )


                # ---------------------------------------------
                # 2. RESIZE + PADDING
                # ---------------------------------------------

                axes[1].imshow(
                    cv.cvtColor(
                        resized,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[1].set_title(
                    "2. Resize + Padding"
                )


                # ---------------------------------------------
                # 3. MULTI-SCALE RETINEX
                # ---------------------------------------------

                axes[2].imshow(
                    cv.cvtColor(
                        corrected,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[2].set_title(
                    "3. MSR"
                )


                # ---------------------------------------------
                # 4. NON-LOCAL MEANS
                # ---------------------------------------------

                axes[3].imshow(
                    cv.cvtColor(
                        denoised,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[3].set_title(
                    "4. NLM"
                )


                # ---------------------------------------------
                # 5. K-MEANS MASK
                # ---------------------------------------------

                axes[4].imshow(
                    mask,
                    cmap="gray"
                )

                axes[4].set_title(
                    "5. K-Means Mask"
                )


                # ---------------------------------------------
                # 6. COLOUR CONSTANCY
                # ---------------------------------------------

                axes[5].imshow(
                    cv.cvtColor(
                        colour_corrected,
                        cv.COLOR_BGR2RGB
                    )
                )

                axes[5].set_title(
                    "6. Colour Constancy"
                )


                # ---------------------------------------------
                # 7. GABOR FILTER
                # ---------------------------------------------

                axes[6].imshow(
                    gabor_img,
                    cmap="gray"
                )

                axes[6].set_title(
                    "7. Gabor"
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
# 11. CONVERT TO NUMPY
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
# 12. DATASET SUMMARY
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
# 13. FEATURE SIZE
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
