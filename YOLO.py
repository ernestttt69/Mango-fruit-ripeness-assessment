from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO


SOURCE_DATASET = Path(r"Z:\mangoDataSet")
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

    plt.show()

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

    device=0,

    workers=0,

    project="mango_yolo_runs",

    name="mango_detector"
)


print(
    "YOLO training completed."
)

best_model_path = Path(detector.trainer.best)

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

    plt.show()

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


# Precision: 0.865 – The model correctly identified 86.5% of the predicted mango detections.
# Recall   : 0.850 – The model successfully detected 85.0% of the actual mango instances.
# mAP@50   : 0.919 – The model achieved 91.9% mean Average Precision at an IoU threshold of 0.50, indicating strong detection performance.
# mAP@50–95: 0.784 – The model achieved 78.4% mean Average Precision across IoU thresholds from 0.50 to 0.95.