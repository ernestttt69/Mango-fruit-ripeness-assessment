import xml.etree.ElementTree as ET
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern
from medpy.filter.smoothing import anisotropic_diffusion

def AGCWD(channel, alpha=0.3):
    hist = cv.calcHist([channel], [0], None, [256], [0,256])
    hist = hist.flatten()
    pdf = hist / np.sum(hist)
    pdf_weighted = pdf ** alpha
    pdf_weighted = pdf_weighted / np.sum(pdf_weighted)
    cdf = np.cumsum(pdf_weighted)
    gamma = 1 - cdf
    LUT = np.zeros(256)
    for i in range(256):
        LUT[i] = 255 * ((i/255) ** gamma[i])
    LUT = np.clip(LUT,0,255).astype(np.uint8)
    enhanced = cv.LUT(channel,LUT)
    return enhanced

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

    for img_path in image_files:
        img = cv.imread(str(img_path))
        if img is None:
            continue

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        xml_path = img_path.with_suffix(".xml")

        if xml_path.exists():
        
            tree = ET.parse(xml_path)
            root = tree.getroot()
        
            for obj in root.findall("object"):
        
                label = obj.find("name").text
        
                b = obj.find("bndbox")
        
                xmin = int(b.find("xmin").text)
                ymin = int(b.find("ymin").text)
                xmax = int(b.find("xmax").text)
                ymax = int(b.find("ymax").text)
        
                crop = img[ymin:ymax, xmin:xmax]
        
                if crop.size == 0:
                    continue
        
                crop = cv.resize(crop, (224, 224))

                crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)

                # Shadow & Illumination Correction
                blur_ambient = cv.GaussianBlur(crop, (15, 15), 0)
                illum_corrected = cv.addWeighted(crop, 1.2, blur_ambient, -0.8, 0)

                # LAB Colour Space
                lab = cv.cvtColor(illum_corrected, cv.COLOR_BGR2LAB)
                L, A, B = cv.split(lab)

                # Adaptive Gamma Correction (AGCWD)
                L_enhanced = AGCWD(L)

                # CLAHE
                clahe = cv.createCLAHE(
                    clipLimit=1.0,
                    tileGridSize=(4, 4)
                )
                L_clahe = clahe.apply(L_enhanced)

                # Merge LAB channels and convert back to BGR
                lab_enhanced = cv.merge([L_clahe, A, B])
                enhanced_bgr = cv.cvtColor(
                    lab_enhanced,
                    cv.COLOR_LAB2BGR
                )
                enhanced_rgb = cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2RGB)

                gray = cv.cvtColor(enhanced_bgr, cv.COLOR_BGR2GRAY)

                # Non-local Means
                denoised = cv.fastNlMeansDenoising(
                    gray,
                    None,
                    h=5,
                    templateWindowSize=3,
                    searchWindowSize=15
                )

                # Adaptive Gaussian Thresholding
                adaptive_mask = cv.adaptiveThreshold(
                    denoised,
                    255,
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv.THRESH_BINARY,
                    7,
                    1
                )

                # Morphological processing
                kernel = np.ones((1, 1), np.uint8)
                adaptive_mask = cv.morphologyEx(
                    adaptive_mask, cv.MORPH_OPEN, kernel
                )
                adaptive_mask = cv.morphologyEx(
                    adaptive_mask, cv.MORPH_CLOSE, kernel
                )

                # Local Feature / Edge Extraction
                radius = 1
            
                n_points = 8 * radius
                    
                segmented = cv.bitwise_and(
                    denoised,
                    denoised,
                    mask=adaptive_mask
                )
                
                lbp = local_binary_pattern(
                    segmented,
                    n_points,
                    radius,
                    method="uniform"
                )

                n_bins = n_points + 2

                hist, _ = np.histogram(lbp.ravel(),bins=n_bins,range=(0, n_bins),density=True)
                
                if split == "train":
                    X_train.append(hist)
                    y_train.append(label)
                
                elif split == "valid":
                    X_valid.append(hist)
                    y_valid.append(label)
                
                elif split == "test":
                    X_test.append(hist)
                    y_test.append(label)

                if display_count < 5:
                    fig, axes = plt.subplots(1, 6, figsize=(21, 3))
                
                    axes[0].imshow(crop_rgb)
                    axes[0].set_title("1. Original")
                    axes[1].imshow(cv.cvtColor(illum_corrected, cv.COLOR_BGR2RGB))
                    axes[1].set_title("2. Illum")
                    axes[2].imshow(enhanced_rgb)
                    axes[2].set_title("3. AGCWD + CLAHE")
                    axes[3].imshow(denoised, cmap="gray")
                    axes[3].set_title("4. Denoised")
                    axes[4].imshow(adaptive_mask, cmap="gray")
                    axes[4].set_title("5. Adaptive")
                    axes[5].imshow(lbp, cmap="gray")
                    axes[5].set_title("6. LBP")
                
                    for ax in axes:
                        ax.axis("off")
                
                    plt.suptitle(f"Split: {split} | File: {img_path.name} | Label : {label}")
                    plt.tight_layout()
                    plt.close(fig)

                    display_count += 1

X_train = np.array(X_train)
y_train = np.array(y_train)

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

X_test = np.array(X_test)
y_test = np.array(y_test)
                
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_valid.shape, y_valid.shape)
print("Testing:", X_test.shape, y_test.shape)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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


from sklearn.metrics import classification_report

print(classification_report(y_test, test_pred))



import matplotlib.pyplot as plt

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

plt.close()


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
plt.close()



from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
 
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred, average="weighted")
recall = recall_score(y_test, test_pred, average="weighted")
f1 = f1_score(y_test, test_pred, average="weighted")

print("----- Final Summary Metrics -----")
print(f"Accuracy : {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall   : {recall:.3f}")
print(f"F1 Score : {f1:.3f}")


import joblib

joblib.dump(svm_model, "mango_svm_model.pkl")
joblib.dump(scaler, "mango_scaler.pkl")

print("\nModel and scaler saved successfully!")

from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import os
import cv2 as cv
import matplotlib.pyplot as plt

from ultralytics import YOLO


SOURCE_DATASET = DATASET_PATH
YOLO_DATASET = Path("./MangoYOLO")

splits = ["train", "valid", "test"]

for split in splits:

    (YOLO_DATASET / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (YOLO_DATASET / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


print("YOLO folder structure created.")

def convert_xml_to_yolo(
    xml_path,
    image_width,
    image_height
):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    yolo_labels = []

    for obj in root.findall("object"):

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(
            bbox.find("xmin").text
        )

        ymin = float(
            bbox.find("ymin").text
        )

        xmax = float(
            bbox.find("xmax").text
        )

        ymax = float(
            bbox.find("ymax").text
        )

        xmin = max(
            0,
            min(
                xmin,
                image_width
            )
        )

        xmax = max(
            0,
            min(
                xmax,
                image_width
            )
        )

        ymin = max(
            0,
            min(
                ymin,
                image_height
            )
        )

        ymax = max(
            0,
            min(
                ymax,
                image_height
            )
        )

        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = (
            (xmin + xmax) / 2
        ) / image_width

        y_center = (
            (ymin + ymax) / 2
        ) / image_height

        box_width = (
            xmax - xmin
        ) / image_width

        box_height = (
            ymax - ymin
        ) / image_height


        class_id = 0


        yolo_labels.append(
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )
    return yolo_labels

for split in splits:

    source_folder = (
        SOURCE_DATASET / split
    )

    if not source_folder.exists():

        print(
            f"Skipping missing folder: "
            f"{source_folder}"
        )

        continue

    image_files = (
        list(
            source_folder.glob("*.jpg")
        )
        +
        list(
            source_folder.glob("*.jpeg")
        )
        +
        list(
            source_folder.glob("*.png")
        )
    )


    print(
        f"\nProcessing {split}: "
        f"{len(image_files)} images"
    )


    converted_count = 0
    skipped_count = 0


    for image_path in image_files:

        xml_path = image_path.with_suffix(
            ".xml"
        )

        if not xml_path.exists():

            print(
                f"Missing XML: "
                f"{image_path.name}"
            )

            skipped_count += 1

            continue

        image = cv.imread(
            str(image_path)
        )


        if image is None:

            print(
                f"Cannot read image: "
                f"{image_path.name}"
            )

            skipped_count += 1

            continue


        height, width = image.shape[:2]

        yolo_labels = convert_xml_to_yolo(
            xml_path,
            width,
            height
        )

        destination_image = (
            YOLO_DATASET
            / "images"
            / split
            / image_path.name
        )


        shutil.copy2(
            image_path,
            destination_image
        )

        destination_label = (
            YOLO_DATASET
            / "labels"
            / split
            / f"{image_path.stem}.txt"
        )


        with open(
            destination_label,
            "w"
        ) as f:

            f.write(
                "\n".join(
                    yolo_labels
                )
            )


        converted_count += 1


    print(
        f"{split}: "
        f"{converted_count} converted, "
        f"{skipped_count} skipped."
    )


print(
    "\nXML to YOLO conversion completed."
)

for split in splits:

    image_folder = (
        YOLO_DATASET
        / "images"
        / split
    )

    label_folder = (
        YOLO_DATASET
        / "labels"
        / split
    )


    image_count = len(
        list(
            image_folder.glob("*.*")
        )
    )

    label_count = len(
        list(
            label_folder.glob("*.txt")
        )
    )


    print(
        f"{split}: "
        f"{image_count} images | "
        f"{label_count} labels"
    )

sample_labels = list(
    (
        YOLO_DATASET
        / "labels"
        / "train"
    ).glob("*.txt")
)


if len(sample_labels) > 0:

    sample_label = sample_labels[0]

    print(
        "\nSample label file:"
    )

    print(
        sample_label
    )

    print(
        "\nContents:"
    )

    with open(
        sample_label,
        "r"
    ) as f:

        print(
            f.read()
        )

train_images = list(
    (
        YOLO_DATASET
        / "images"
        / "train"
    ).glob("*.jpg")
)


if len(train_images) > 0:

    sample_image_path = train_images[0]

    sample_label_path = (
        YOLO_DATASET
        / "labels"
        / "train"
        / f"{sample_image_path.stem}.txt"
    )


    image = cv.imread(
        str(
            sample_image_path
        )
    )


    height, width = image.shape[:2]


    if sample_label_path.exists():

        with open(
            sample_label_path,
            "r"
        ) as f:

            lines = f.readlines()


        for line in lines:

            values = line.strip().split()

            if len(values) != 5:
                continue


            class_id = int(
                values[0]
            )

            x_center = float(
                values[1]
            )

            y_center = float(
                values[2]
            )

            box_width = float(
                values[3]
            )

            box_height = float(
                values[4]
            )


            # Convert YOLO back
            # into pixel coordinates
            x_center *= width
            y_center *= height

            box_width *= width
            box_height *= height


            xmin = int(
                x_center
                -
                box_width / 2
            )

            ymin = int(
                y_center
                -
                box_height / 2
            )

            xmax = int(
                x_center
                +
                box_width / 2
            )

            ymax = int(
                y_center
                +
                box_height / 2
            )


            cv.rectangle(
                image,
                (xmin, ymin),
                (xmax, ymax),
                (0, 255, 0),
                2
            )


            cv.putText(
                image,
                "mango",
                (
                    xmin,
                    max(
                        ymin - 10,
                        20
                    )
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cv.cvtColor(
            image,
            cv.COLOR_BGR2RGB
        )
    )

    plt.title(
        "YOLO Annotation Check"
    )

    plt.axis(
        "off"
    )

    plt.close()

dataset_root = (
    YOLO_DATASET.resolve()
)


yaml_content = f"""
path: {dataset_root.as_posix()}

train: images/train
val: images/valid
test: images/test

names:
  0: mango
"""


with open(
    "mango.yaml",
    "w"
) as f:

    f.write(
        yaml_content
    )


print(
    "\nmango.yaml created successfully:"
)

print(
    yaml_content
)

detector = YOLO(
    "yolov8n.pt"
)


print(
    "YOLO base model loaded."
)

training_results = detector.train(

    data="mango.yaml",

    epochs=50,

    imgsz=640,

    batch=8,

    workers=0,

    project="mango_yolo_runs",

    name="mango_detector"
)


print(
    "YOLO training completed."
)

best_model_path = Path(training_results.save_dir) / "weights" / "best.pt"

if best_model_path.exists():

    best_model = YOLO(
        str(
            best_model_path
        )
    )

    print(
        "Best YOLO detector loaded:"
    )

    print(
        best_model_path
    )

else:

    print(
        "best.pt not found."
    )

test_images = (
    list(
        (
            YOLO_DATASET
            / "images"
            / "test"
        ).glob("*.jpg")
    )
    +
    list(
        (
            YOLO_DATASET
            / "images"
            / "test"
        ).glob("*.jpeg")
    )
    +
    list(
        (
            YOLO_DATASET
            / "images"
            / "test"
        ).glob("*.png")
    )
)


if (
    best_model_path.exists()
    and len(test_images) > 0
):

    test_image_path = (
        test_images[0]
    )


    results = best_model.predict(

        source=str(
            test_image_path
        ),

        conf=0.5,

        save=False
    )


    plotted_image = (
        results[0].plot()
    )


    plt.figure(
        figsize=(10, 8)
    )

    plt.imshow(
        cv.cvtColor(
            plotted_image,
            cv.COLOR_BGR2RGB
        )
    )

    plt.title(
        "YOLO Mango Detection Result"
    )

    plt.axis(
        "off"
    )

    plt.close()

if best_model_path.exists():

    destination_model = Path(
        "mango_detector.pt"
    )


    shutil.copy2(
        best_model_path,
        destination_model
    )


    print(
        "\nMango detector saved as:"
    )

    print(
        destination_model.resolve()
    )

else:

    print(
        "Unable to create mango_detector.pt "
        "because best.pt was not found."
    )
