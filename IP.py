import xml.etree.ElementTree as ET
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

cv.setUseOptimized(True)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ==========================================================
# Data Preprocessing 1:
# Contrast Limited Adaptive Gamma Correction (CLAGC)
# Enhances local contrast and improves image brightness.
# ==========================================================
def CLAGC(
    channel,
    clip_limit=2.0,
    tile_grid_size=(8, 8),
    gamma=0.8,
):
    # Enhance local contrast using CLAHE
    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    contrast_enhanced = clahe.apply(channel)

    # Adjust image brightness using gamma correction
    normalized = contrast_enhanced.astype(np.float32) / 255.0
    gamma_corrected = np.power(normalized, gamma)
    result = gamma_corrected * 255.0

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================================
# Data Preprocessing 2:
# Retinex-Based Colour Correction
# Corrects uneven illumination in one colour channel.
# ==========================================================
def single_scale_retinex(channel, sigma=30):
    # Convert to float and avoid logarithm of zero
    channel_float = channel.astype(np.float32) + 1.0

    # Estimate the illumination component
    blurred = cv.GaussianBlur(
        channel_float,
        (0, 0),
        sigma,
    )

    # Separate reflectance from illumination
    retinex = np.log(channel_float) - np.log(blurred + 1.0)

    # Convert the result back to an 8-bit image
    retinex = cv.normalize(
        retinex,
        None,
        0,
        255,
        cv.NORM_MINMAX,
    )

    return retinex.astype(np.uint8)


# ==========================================================
# Data Preprocessing 2:
# Retinex-Based Colour Correction
# Applies Retinex correction to each colour channel.
# ==========================================================
def retinex_colour_correction(image, sigma=80):
    # Split the BGR image into separate colour channels
    blue, green, red = cv.split(image)

    # Apply Retinex correction to each channel
    blue_corrected = single_scale_retinex(blue, sigma)
    green_corrected = single_scale_retinex(green, sigma)
    red_corrected = single_scale_retinex(red, sigma)

    # Merge the corrected channels
    corrected_image = cv.merge(
        [blue_corrected, green_corrected, red_corrected]
    )

    corrected_image = cv.addWeighted(
    image,
    0.7,
    corrected_image,
    0.3,
    0
)

    return corrected_image


# ==========================================================
# Data Preprocessing 4:
# K-Means Clustering Segmentation
# Separates the mango region from the background.
# ==========================================================
def kmeans_segmentation(image, k=3):
    # Reshape the image into a list of pixels
    height, width = image.shape[:2]
    scale = min(1.0, 128.0 / max(height, width))

    if scale < 1.0:
        clustering_image = cv.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv.INTER_AREA,
        )
    else:
        clustering_image = image

    clustering_height, clustering_width = clustering_image.shape[:2]
    lab_image = cv.cvtColor(clustering_image, cv.COLOR_BGR2LAB)

    y_coordinates, x_coordinates = np.indices(
        (clustering_height, clustering_width),
        dtype=np.float32,
    )
    x_coordinates = x_coordinates / max(1, clustering_width - 1) * 255.0
    y_coordinates = y_coordinates / max(1, clustering_height - 1) * 255.0

    pixel_values = np.column_stack(
        [
            lab_image.reshape((-1, 3)).astype(np.float32),
            (x_coordinates.reshape(-1) * 0.20),
            (y_coordinates.reshape(-1) * 0.20),
        ]
    )

    # Define the K-Means stopping criteria
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        25,
        0.5,
    )

    # Group pixels into colour clusters
    cv.setRNGSeed(42)
    _, labels, centers = cv.kmeans(
        pixel_values,
        k,
        None,
        criteria,
        2,
        cv.KMEANS_PP_CENTERS,
    )

    labels = labels.flatten()
    centers = centers.astype(np.float32)
    label_image = labels.reshape((clustering_height, clustering_width))

    # Select the cluster at the image centre
    center_y = clustering_height // 2
    center_x = clustering_width // 2

    center_mask = np.zeros(
        (clustering_height, clustering_width),
        dtype=np.uint8,
    )
    cv.ellipse(
        center_mask,
        (center_x, center_y),
        (
            max(1, int(clustering_width * 0.30)),
            max(1, int(clustering_height * 0.34)),
        ),
        0,
        0,
        360,
        255,
        -1,
    )

    border_size = max(2, int(min(clustering_height, clustering_width) * 0.08))
    border_mask = np.zeros(
        (clustering_height, clustering_width),
        dtype=np.uint8,
    )
    border_mask[:border_size, :] = 255
    border_mask[-border_size:, :] = 255
    border_mask[:, :border_size] = 255
    border_mask[:, -border_size:] = 255

    center_pixels = center_mask > 0
    border_pixels = border_mask > 0
    center_total = max(1, int(np.count_nonzero(center_pixels)))
    border_total = max(1, int(np.count_nonzero(border_pixels)))

    cluster_scores = []

    for cluster_index in range(k):
        cluster_pixels = label_image == cluster_index
        center_share = np.count_nonzero(cluster_pixels & center_pixels) / center_total
        border_share = np.count_nonzero(cluster_pixels & border_pixels) / border_total
        area_share = np.count_nonzero(cluster_pixels) / label_image.size
        score = (2.5 * center_share) - (1.5 * border_share) + (0.15 * area_share)
        cluster_scores.append(score)

    cluster_scores = np.asarray(cluster_scores, dtype=np.float32)
    mango_cluster = int(np.argmax(cluster_scores))

    selected_clusters = [mango_cluster]
    best_score = float(cluster_scores[mango_cluster])
    best_colour = centers[mango_cluster, :3]

    for cluster_index in range(k):
        if cluster_index == mango_cluster:
            continue

        cluster_pixels = label_image == cluster_index
        center_share = np.count_nonzero(cluster_pixels & center_pixels) / center_total
        border_share = np.count_nonzero(cluster_pixels & border_pixels) / border_total
        colour_distance = np.linalg.norm(centers[cluster_index, :3] - best_colour)

        if (
            center_share >= 0.10
            and border_share <= 0.45
            and cluster_scores[cluster_index] >= best_score * 0.20
            and colour_distance <= 95.0
        ):
            selected_clusters.append(cluster_index)

    # Create a binary mask for the selected cluster
    mask = np.isin(label_image, selected_clusters).astype(np.uint8) * 255

    component_count, component_labels, component_stats, _ = cv.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if component_count > 1:
        best_component = 0
        best_component_score = -1.0

        for component_index in range(1, component_count):
            component_pixels = component_labels == component_index
            component_area = component_stats[component_index, cv.CC_STAT_AREA]
            center_overlap = np.count_nonzero(component_pixels & center_pixels)
            component_score = center_overlap * 4.0 + component_area

            if component_score > best_component_score:
                best_component_score = component_score
                best_component = component_index

        mask = np.where(
            component_labels == best_component,
            255,
            0,
        ).astype(np.uint8)

    if mask.shape != image.shape[:2]:
        mask = cv.resize(
            mask,
            (width, height),
            interpolation=cv.INTER_NEAREST,
        )

    # Remove small noise and fill small gaps in the mask
    kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel_close)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_open)

    flood_fill_mask = np.zeros((height + 2, width + 2), np.uint8)
    filled_mask = mask.copy()
    cv.floodFill(filled_mask, flood_fill_mask, (0, 0), 255)
    holes = cv.bitwise_not(filled_mask)
    mask = cv.bitwise_or(mask, holes)

    foreground_ratio = np.count_nonzero(mask) / mask.size

    if foreground_ratio < 0.25 or foreground_ratio > 0.95:
        fallback_mask = np.zeros((height, width), dtype=np.uint8)
        cv.ellipse(
            fallback_mask,
            (width // 2, height // 2),
            (
                max(1, int(width * 0.43)),
                max(1, int(height * 0.47)),
            ),
            0,
            0,
            360,
            255,
            -1,
        )
        mask = fallback_mask

    mask = cv.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv.threshold(mask, 127, 255, cv.THRESH_BINARY)

    # Apply the segmentation mask to the image
    segmented_image = cv.bitwise_and(
        image,
        image,
        mask=mask,
    )

    return segmented_image, mask


# ==========================================================
# Data Preprocessing 5:
# Homomorphic Filtering
# Reduces illumination variations and enhances image details.
# ==========================================================
_homomorphic_filter_cache = {}


def homomorphic_filter(
    gray_image,
    gamma_low=0.8,
    gamma_high=1.2,
    cutoff=60,
):
    # Convert to float and apply logarithmic transformation
    gray_float = gray_image.astype(np.float32) + 1.0
    log_image = np.log(gray_float)

    rows, cols = gray_float.shape
    padded_rows = cv.getOptimalDFTSize(rows)
    padded_cols = cv.getOptimalDFTSize(cols)

    # Pad the image for frequency-domain processing
    padded_image = np.zeros(
        (padded_rows, padded_cols),
        dtype=np.float32,
    )
    padded_image[:rows, :cols] = log_image

    # Transform the image into the frequency domain
    frequency_image = np.fft.fft2(padded_image)
    frequency_image = np.fft.fftshift(frequency_image)

    # Calculate the distance from the frequency centre
    cache_key = (
        padded_rows,
        padded_cols,
        float(gamma_low),
        float(gamma_high),
        float(cutoff),
    )
    filter_mask = _homomorphic_filter_cache.get(cache_key)

    # Create a homomorphic high-pass filter
    if filter_mask is None:
        y, x = np.indices((padded_rows, padded_cols), dtype=np.float32)
        center_y = padded_rows // 2
        center_x = padded_cols // 2
        distance_squared = (
            (x - center_x) ** 2
            + (y - center_y) ** 2
        )

        filter_mask = (
            (gamma_high - gamma_low)
            * (1.0 - np.exp(-distance_squared / (2.0 * cutoff**2)))
            + gamma_low
        ).astype(np.float32)
        _homomorphic_filter_cache[cache_key] = filter_mask

    # Apply the filter in the frequency domain
    filtered_frequency = frequency_image * filter_mask

    # Transform the result back to the spatial domain
    filtered_frequency = np.fft.ifftshift(filtered_frequency)
    filtered_image = np.fft.ifft2(filtered_frequency)
    filtered_image = np.real(filtered_image)

    # Reverse the logarithmic transformation
    filtered_image = np.exp(filtered_image) - 1.0
    filtered_image = filtered_image[:rows, :cols]

    # Normalize the result to the 0-255 range
    filtered_image = cv.normalize(
        filtered_image,
        None,
        0,
        255,
        cv.NORM_MINMAX,
    )

    return filtered_image.astype(np.uint8)


# ==========================================================
# Feature Extraction 6:
# Gabor Filter
# Extracts texture features for SVM classification.
# ==========================================================
def extract_gabor_features(
    image,
    frequencies=(0.05, 0.1, 0.15, 0.2, 0.3),
    orientations=(
        0,
        np.pi / 8,
        np.pi / 4,
        3 * np.pi / 8,
        np.pi / 2,
        5 * np.pi / 8,
        3 * np.pi / 4,
        7 * np.pi / 8,
    ),
):
    features = []
    responses = []

    # Convert the image to the 0-1 floating-point range
    image_float = image.astype(np.float32) / 255.0

    # Apply Gabor filters at different frequencies and directions
    for frequency in frequencies:
        wavelength = 1.0 / frequency

        for theta in orientations:
            gabor_kernel = cv.getGaborKernel(
                ksize=(31, 31),
                sigma=6.0,
                theta=theta,
                lambd=wavelength,
                gamma=0.8,
                psi=0,
                ktype=cv.CV_32F,
            )

            response = cv.filter2D(
                image_float,
                cv.CV_32F,
                gabor_kernel,
            )

            # Store the mean and standard deviation as features
            features.append(np.mean(response))
            features.append(np.std(response))
            responses.append(response)

    feature_vector = np.array(features, dtype=np.float32)

    return feature_vector, responses


DATASET_PATH = Path(r"Z:\mangoDataset")
SPLITS = ["train", "valid", "test"]

X_train = []
y_train = []
X_valid = []
y_valid = []
X_test = []
y_test = []

display_count = 0

for split in SPLITS:
    folder = DATASET_PATH / split

    if not folder.exists():
        print(f"Skipping missing folder: {folder}")
        continue

    image_files = list(folder.glob("*.jpg"))
    print(f"\nFound {len(image_files)} images in '{split}' set.")

    for i, img_path in enumerate(image_files, 1):

        if i % 100 == 0:
            print(f"{split}: {i}/{len(image_files)}")

        img = cv.imread(str(img_path))

        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue

        xml_path = img_path.with_suffix(".xml")

        if not xml_path.exists():
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label_element = obj.find("name")
            box_element = obj.find("bndbox")

            if label_element is None or box_element is None:
                continue

            label = label_element.text

            xmin = int(float(box_element.find("xmin").text))
            ymin = int(float(box_element.find("ymin").text))
            xmax = int(float(box_element.find("xmax").text))
            ymax = int(float(box_element.find("ymax").text))

            # Keep the bounding box inside the image dimensions
            height, width = img.shape[:2]
            xmin = max(0, min(xmin, width - 1))
            xmax = max(0, min(xmax, width))
            ymin = max(0, min(ymin, height - 1))
            ymax = max(0, min(ymax, height))

            crop = img[ymin:ymax, xmin:xmax]

            if crop.size == 0:
                continue

            crop = cv.resize(crop, (160, 160), interpolation=cv.INTER_AREA)
            crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)

            # --------------------------------------------------
            # 1. Contrast Limited Adaptive Gamma Correction
            # Enhance local contrast and adjust image brightness.
            # --------------------------------------------------
            hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv)

            v_clagc = CLAGC(
                v,
                clip_limit=1.8,
                tile_grid_size=(8, 8),
                gamma=0.95,
            )

            clagc_hsv = cv.merge([h, s, v_clagc])
            clagc_bgr = cv.cvtColor(clagc_hsv, cv.COLOR_HSV2BGR)
            clagc_rgb = cv.cvtColor(clagc_bgr, cv.COLOR_BGR2RGB)

            # --------------------------------------------------
            # 2. Retinex-Based Colour Correction
            # Correct uneven illumination and colour appearance.
            # --------------------------------------------------
            retinex_bgr = retinex_colour_correction(
                clagc_bgr,
                sigma=60,
            )
            retinex_rgb = cv.cvtColor(retinex_bgr, cv.COLOR_BGR2RGB)

            # --------------------------------------------------
            # 3. Non-Local Means Denoising
            # Remove image noise while preserving image details.
            # --------------------------------------------------
            denoised_bgr = cv.fastNlMeansDenoisingColored(
                retinex_bgr,
                None,
                h=2,
                hColor=2,
                templateWindowSize=7,
                searchWindowSize=7,
            )
            denoised_rgb = cv.cvtColor(denoised_bgr, cv.COLOR_BGR2RGB)

            # --------------------------------------------------
            # 4. K-Means Clustering Segmentation
            # Separate the mango region from the background.
            # --------------------------------------------------
            segmented_bgr, kmeans_mask = kmeans_segmentation(
                denoised_bgr,
                k=3,
            )

            # --------------------------------------------------
            # 5. Homomorphic Filtering
            # Reduce lighting variations and enhance details.
            # --------------------------------------------------
            segmented_gray = cv.cvtColor(
                segmented_bgr,
                cv.COLOR_BGR2GRAY,
            )

            homomorphic = homomorphic_filter(
                segmented_gray,
                gamma_low=0.90,
                gamma_high=1.10,
                cutoff=55,
            )

            # --------------------------------------------------
            # 6. Gabor Filter Feature Extraction
            # Extract texture features for SVM classification.
            # --------------------------------------------------
            features, gabor_responses = extract_gabor_features(
                homomorphic
            )

            if split == "train":
                X_train.append(features)
                y_train.append(label)
            elif split == "valid":
                X_valid.append(features)
                y_valid.append(label)
            elif split == "test":
                X_test.append(features)
                y_test.append(label)

            # Display the first five preprocessing examples
            if display_count < 5:
                fig, axes = plt.subplots(1, 7, figsize=(21, 3))

                axes[0].imshow(crop_rgb)
                axes[0].set_title("1. Original")

                axes[1].imshow(clagc_rgb)
                axes[1].set_title("2. CLAGC")

                axes[2].imshow(retinex_rgb)
                axes[2].set_title("3. Retinex")

                axes[3].imshow(denoised_rgb)
                axes[3].set_title("4. NLM")

                axes[4].imshow(kmeans_mask, cmap="gray")
                axes[4].set_title("5. K-Means")

                axes[5].imshow(homomorphic, cmap="gray")
                axes[5].set_title("6. Homomorphic")

                axes[6].imshow(max(gabor_responses, key=lambda response: np.std(response)), cmap="gray")
                axes[6].set_title("7. Gabor")

                for ax in axes:
                    ax.axis("off")

                plt.suptitle(
                    f"Split: {split} | "
                    f"File: {img_path.name} | "
                    f"Label: {label}"
                )
                plt.tight_layout()
                plt.show()

                display_count += 1


X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train)
X_valid = np.asarray(X_valid, dtype=np.float32)
y_valid = np.asarray(y_valid)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test)

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_valid.shape, y_valid.shape)
print("Testing:", X_test.shape, y_test.shape)

if X_train.size == 0:
    raise ValueError("No training features were extracted. Check the dataset path and XML files.")

if X_valid.size == 0:
    raise ValueError("No validation features were extracted. Check the valid folder and XML files.")

if X_test.size == 0:
    raise ValueError("No testing features were extracted. Check the test folder and XML files.")


# Scale the extracted Gabor features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


# Train the SVM classifier
svm_model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42,
)
svm_model.fit(X_train_scaled, y_train)


# Generate predictions
train_pred = svm_model.predict(X_train_scaled)
valid_pred = svm_model.predict(X_valid_scaled)
test_pred = svm_model.predict(X_test_scaled)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Validation Accuracy:", accuracy_score(y_valid, valid_pred))
print("Testing Accuracy:", accuracy_score(y_test, test_pred))


# Display training and testing accuracy
train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

plt.figure(figsize=(5, 5))
plt.bar(
    ["Training", "Testing"],
    [train_acc, test_acc],
)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("SVM Classification Accuracy")
plt.tight_layout()
plt.show()


# Display the confusion matrix using a fixed class order
class_labels = [
    "Unripe",
    "Partially Ripe",
    "Ripe",
    "Over Ripe",
]

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=class_labels,
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Final Results Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.show()


# Calculate final evaluation metrics
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(
    y_test,
    test_pred,
    average="weighted",
    zero_division=0,
)
recall = recall_score(
    y_test,
    test_pred,
    average="weighted",
    zero_division=0,
)
f1 = f1_score(
    y_test,
    test_pred,
    average="weighted",
    zero_division=0,
)

print("----- Final Summary Metrics -----")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")