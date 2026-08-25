import streamlit as st
import joblib
import cv2 as cv
import numpy as np

from ultralytics import YOLO
from mango_processing import process_mango_image


# Page configuration
st.set_page_config(
    page_title="Mango Ripeness Detection",
    page_icon="🥭",
    layout="wide"
)

st.title("🥭 Mango Ripeness Detection System")

st.write(
    "Upload a mango image to detect individual mangoes, "
    "analyse their image processing stages, and classify "
    "the ripeness of each mango."
)


# Load trained models
@st.cache_resource
def load_models():

    svm_model = joblib.load(
        "mango_svm_model.pkl"
    )

    scaler = joblib.load(
        "mango_scaler.pkl"
    )

    detector = YOLO(
        "mango_detector.pt"
    )

    return (
        svm_model,
        scaler,
        detector
    )


try:

    svm_model, scaler, detector = load_models()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# Convert BGR image to RGB
def bgr_to_rgb(image):

    if image is None:
        return None

    return cv.cvtColor(
        image,
        cv.COLOR_BGR2RGB
    )


# Check uploaded image quality
def check_image_quality(image_bgr):

    gray = cv.cvtColor(
        image_bgr,
        cv.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray)
    )

    if brightness < 60:

        brightness_status = "Too Dark"

    elif brightness > 210:

        brightness_status = "Too Bright"

    else:

        brightness_status = "Good"

    sharpness = float(
        cv.Laplacian(
            gray,
            cv.CV_64F
        ).var()
    )

    if sharpness < 50:

        sharpness_status = "Blurry"

    elif sharpness < 100:

        sharpness_status = "Slightly Blurry"

    else:

        sharpness_status = "Good"

    height, width = image_bgr.shape[:2]

    if (
        width < 224
        or
        height < 224
    ):

        resolution_status = "Low"

    else:

        resolution_status = "Good"

    if (
        brightness_status != "Good"
        or
        sharpness_status == "Blurry"
        or
        resolution_status == "Low"
    ):

        overall = "Poor"

    else:

        overall = "Acceptable"

    return {

        "brightness":
            brightness,

        "brightness_status":
            brightness_status,

        "sharpness":
            sharpness,

        "sharpness_status":
            sharpness_status,

        "width":
            width,

        "height":
            height,

        "resolution_status":
            resolution_status,

        "overall":
            overall
    }


# Predict mango ripeness using SVM
def predict_mango(image_bgr):

    result = process_mango_image(
        image_bgr
    )

    features = np.asarray(
        result["features"],
        dtype=np.float32
    ).reshape(
        1,
        -1
    )

    scaled_features = scaler.transform(
        features
    )

    prediction = svm_model.predict(
        scaled_features
    )[0]

    confidence = None

    if hasattr(
        svm_model,
        "predict_proba"
    ):

        probabilities = svm_model.predict_proba(
            scaled_features
        )[0]

        print("\nSVM probabilities:")

        for class_name, probability in zip(
            svm_model.classes_,
            probabilities
        ):
            print(
                f"{class_name}: "
                f"{probability * 100:.2f}%"
            )

        confidence = (
            float(
                np.max(probabilities)
            )
            * 100
        )

    return (
        result,
        prediction,
        confidence
    )


# Calculate IoU between two YOLO boxes
def calculate_iou(box1, box2):

    x1 = max(
        box1["x1"],
        box2["x1"]
    )

    y1 = max(
        box1["y1"],
        box2["y1"]
    )

    x2 = min(
        box1["x2"],
        box2["x2"]
    )

    y2 = min(
        box1["y2"],
        box2["y2"]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection_area = (
        intersection_width
        *
        intersection_height
    )

    area1 = (
        (box1["x2"] - box1["x1"])
        *
        (box1["y2"] - box1["y1"])
    )

    area2 = (
        (box2["x2"] - box2["x1"])
        *
        (box2["y2"] - box2["y1"])
    )

    union_area = (
        area1
        +
        area2
        -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        /
        union_area
    )


# Remove duplicate YOLO detections
def remove_duplicate_boxes(
    boxes,
    iou_threshold=0.50
):

    if len(boxes) == 0:
        return []

    # Highest YOLO confidence first
    boxes_sorted = sorted(
        boxes,
        key=lambda x: x["confidence"],
        reverse=True
    )

    unique_boxes = []

    for current in boxes_sorted:

        is_duplicate = False

        for kept in unique_boxes:

            iou = calculate_iou(
                current,
                kept
            )

            if iou >= iou_threshold:

                is_duplicate = True
                break

        if not is_duplicate:

            unique_boxes.append(
                current
            )

    return unique_boxes


# Detect mangoes using YOLO and classify using SVM
def detect_and_classify_mangoes(
    image_bgr,
    detection_threshold=0.40
):

    output_image = image_bgr.copy()

    detections = detector(
        image_bgr,
        conf=detection_threshold,
        iou=0.4,
        verbose=False
    )

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    image_area = (
        image_width
        *
        image_height
    )

    candidate_boxes = []

    # Collect all YOLO candidate boxes
    for detection_result in detections:

        if detection_result.boxes is None:
            continue

        for box in detection_result.boxes:

            detection_confidence = float(
                box.conf[0]
            )

            if (
                detection_confidence
                <
                detection_threshold
            ):
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            x1 = max(
                0,
                min(
                    x1,
                    image_width - 1
                )
            )

            y1 = max(
                0,
                min(
                    y1,
                    image_height - 1
                )
            )

            x2 = max(
                0,
                min(
                    x2,
                    image_width
                )
            )

            y2 = max(
                0,
                min(
                    y2,
                    image_height
                )
            )

            if (
                x2 <= x1
                or
                y2 <= y1
            ):
                continue

            box_width = (
                x2 - x1
            )

            box_height = (
                y2 - y1
            )

            box_area = (
                box_width
                *
                box_height
            )

            area_ratio = (
                box_area
                /
                image_area
            )

            candidate_boxes.append({

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "width":
                    box_width,

                "height":
                    box_height,

                "area":
                    box_area,

                "area_ratio":
                    area_ratio,

                "confidence":
                    detection_confidence
            })


    filtered_boxes = []

    # Remove suspicious background and table detections
    for i, current in enumerate(
        candidate_boxes
    ):

        contained_boxes = 0

        current_area = (
            current["area"]
        )

        # Count smaller detections inside the current box
        for j, other in enumerate(
            candidate_boxes
        ):

            if i == j:
                continue

            other_center_x = (
                other["x1"]
                +
                other["x2"]
            ) / 2

            other_center_y = (
                other["y1"]
                +
                other["y2"]
            ) / 2

            centre_inside = (
                current["x1"]
                <=
                other_center_x
                <=
                current["x2"]

                and

                current["y1"]
                <=
                other_center_y
                <=
                current["y2"]
            )

            much_smaller = (
                other["area"]
                <
                current_area * 0.50
            )

            if (
                centre_inside
                and
                much_smaller
            ):

                contained_boxes += 1


        # Count image borders touched by the box
        border_margin = 15

        touches_left = (
            current["x1"]
            <=
            border_margin
        )

        touches_top = (
            current["y1"]
            <=
            border_margin
        )

        touches_right = (
            current["x2"]
            >=
            image_width - border_margin
        )

        touches_bottom = (
            current["y2"]
            >=
            image_height - border_margin
        )

        border_touch_count = sum([
            touches_left,
            touches_top,
            touches_right,
            touches_bottom
        ])


        # Calculate the shape of the detected box
        aspect_ratio = (
            current["width"]
            /
            max(
                current["height"],
                1
            )
        )


        # Reject large boxes containing several smaller detections
        if (
            len(candidate_boxes) > 1
            and
            current["area_ratio"] > 0.30
            and
            contained_boxes >= 2
        ):
            continue


        # Reject large boxes touching several image borders
        if (
            len(candidate_boxes) > 1
            and
            current["area_ratio"] > 0.30
            and
            border_touch_count >= 2
        ):
            continue


        # Reject large low-confidence detections in multi-object images
        if (
            len(candidate_boxes) > 1
            and
            current["area_ratio"] > 0.25
            and
            current["confidence"] < 0.50
        ):
            continue


        # Reject unrealistic detection shapes
        if (
            aspect_ratio < 0.20
            or
            aspect_ratio > 3.0
        ):
            continue


        filtered_boxes.append(
            current
        )


    # Remove duplicate overlapping detections
    filtered_boxes = remove_duplicate_boxes(
        filtered_boxes,
        iou_threshold=0.50
    )


    # Sort mangoes from left to right
    filtered_boxes = sorted(
        filtered_boxes,
        key=lambda box: (
            box["x1"],
            box["y1"]
        )
    )


    mango_results = []

    mango_id = 1


    # Classify each remaining mango
    for box_info in filtered_boxes:

        x1 = box_info["x1"]
        y1 = box_info["y1"]
        x2 = box_info["x2"]
        y2 = box_info["y2"]

        detection_confidence = (
            box_info["confidence"]
        )


        # Add padding around the detected mango
        padding = 10

        x1_pad = max(
            0,
            x1 - padding
        )

        y1_pad = max(
            0,
            y1 - padding
        )

        x2_pad = min(
            image_width,
            x2 + padding
        )

        y2_pad = min(
            image_height,
            y2 + padding
        )


        # Crop the mango ROI
        mango_roi = image_bgr[
            y1_pad:y2_pad,
            x1_pad:x2_pad
        ]


        if (
            mango_roi is None
            or
            mango_roi.size == 0
        ):
            continue


        # Classify ripeness using the SVM
        (
            processing_result,
            prediction,
            confidence
        ) = predict_mango(
            mango_roi
        )


        if confidence is not None:
            label = (
                f"Mango {mango_id}: "
                f"{prediction} "
                f"{confidence:.0f}%"
            )
        
        else:
            label = (
                f"Mango {mango_id}: "
                f"{prediction}"
            )

        # Draw the mango bounding box
        cv.rectangle(
            output_image,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (
                0,
                255,
                0
            ),
            3
        )


        font = (
            cv.FONT_HERSHEY_SIMPLEX
        )

        font_scale = 0.55

        thickness = 2


        (
            text_width,
            text_height
        ), _ = cv.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )


        label_y = max(
            y1 - 10,
            text_height + 10
        )


        label_x2 = min(
            x1
            +
            text_width
            +
            8,
            image_width
        )


        # Draw the label background
        cv.rectangle(
            output_image,
            (
                x1,
                label_y
                -
                text_height
                -
                8
            ),
            (
                label_x2,
                label_y + 4
            ),
            (
                0,
                255,
                0
            ),
            -1
        )


        # Draw the label text
        cv.putText(
            output_image,
            label,
            (
                x1 + 4,
                label_y
            ),
            font,
            font_scale,
            (
                0,
                0,
                0
            ),
            thickness
        )


        mango_results.append({

            "id":
                mango_id,

            "class":
                prediction,

            "confidence":
                confidence,

            "detection_confidence":
                detection_confidence,

            "roi":
                mango_roi,

            "processing":
                processing_result,

            "box":
                (
                    x1,
                    y1,
                    x2,
                    y2
                )
        })


        mango_id += 1


    return (
        output_image,
        mango_results
    )


# Initialise Streamlit session state
if (
    "uploader_key"
    not in st.session_state
):

    st.session_state.uploader_key = 0


if (
    "mango_results"
    not in st.session_state
):

    st.session_state.mango_results = None


if (
    "annotated_image"
    not in st.session_state
):

    st.session_state.annotated_image = None


if (
    "analysed_filename"
    not in st.session_state
):

    st.session_state.analysed_filename = None


# Upload mango image
uploaded_image = st.file_uploader(
    "Upload Mango Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key=(
        f"mango_uploader_"
        f"{st.session_state.uploader_key}"
    )
)


if uploaded_image is not None:


    # Decode the uploaded image
    file_bytes = np.frombuffer(
        uploaded_image.getvalue(),
        dtype=np.uint8
    )


    image_bgr = cv.imdecode(
        file_bytes,
        cv.IMREAD_COLOR
    )


    if image_bgr is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()


    # Clear results when a different image is uploaded
    if (
        st.session_state.analysed_filename
        is not None
        and
        st.session_state.analysed_filename
        != uploaded_image.name
    ):

        st.session_state.mango_results = None

        st.session_state.annotated_image = None

        st.session_state.analysed_filename = None


        if (
            "selected_mango"
            in st.session_state
        ):

            del st.session_state[
                "selected_mango"
            ]


    # Display uploaded image
    st.subheader(
        "📷 Uploaded Image"
    )


    st.image(
        bgr_to_rgb(
            image_bgr
        ),
        caption=(
            "Input Mango Image"
        ),
        use_container_width=True
    )


    # Assess image quality
    quality = check_image_quality(
        image_bgr
    )


    st.subheader(
        "🖼️ Image Quality Assessment"
    )


    q1, q2, q3 = st.columns(3)


    with q1:

        st.metric(
            "Brightness",
            quality[
                "brightness_status"
            ],
            f'{quality["brightness"]:.1f}'
        )


    with q2:

        st.metric(
            "Sharpness",
            quality[
                "sharpness_status"
            ],
            f'{quality["sharpness"]:.1f}'
        )


    with q3:

        st.metric(
            "Resolution",
            quality[
                "resolution_status"
            ],
            (
                f'{quality["width"]}'
                f' × '
                f'{quality["height"]}'
            )
        )


    if (
        quality["overall"]
        ==
        "Poor"
    ):

        st.warning(
            "⚠️ Poor image quality detected. "
            "This may affect detection and "
            "classification accuracy."
        )

    else:

        st.success(
            "✅ Image quality is acceptable "
            "for analysis."
        )


    # Analyse the uploaded image
    if st.button(
        "Analyse Mango",
        type="primary",
        key="analyse_mango"
    ):

        try:

            with st.spinner(
                "Detecting and classifying mangoes..."
            ):

                (
                    annotated_image,
                    mango_results
                ) = detect_and_classify_mangoes(
                    image_bgr
                )


            st.session_state.annotated_image = (
                annotated_image
            )


            st.session_state.mango_results = (
                mango_results
            )


            st.session_state.analysed_filename = (
                uploaded_image.name
            )


            if (
                "selected_mango"
                in st.session_state
            ):

                del st.session_state[
                    "selected_mango"
                ]


            st.rerun()


        except Exception as e:

            st.error(
                f"Image processing error: "
                f"{type(e).__name__}: {e}"
            )


    # Display saved analysis results
    if (
        st.session_state.mango_results
        is not None
    ):

        mango_results = (
            st.session_state.mango_results
        )


        annotated_image = (
            st.session_state.annotated_image
        )


        st.divider()


        st.header(
            "🥭 Individual Mango Detection"
        )


        if len(mango_results) == 0:

            st.warning(
                "⚠️ No mango was detected "
                "in the image."
            )


        else:

            st.image(
                bgr_to_rgb(
                    annotated_image
                ),
                caption=(
                    "Detected Mangoes "
                    "with Ripeness Labels"
                ),
                use_container_width=True
            )


            st.write(
                f"**Total mangoes detected: "
                f"{len(mango_results)}**"
            )


            # Display all classification results
            st.divider()


            st.header(
                "🥭 All Mango Classification Results"
            )


            for mango in mango_results:

                st.subheader(
                    f"Mango {mango['id']}"
                )


                c1, c2, c3, c4 = (
                    st.columns(4)
                )


                with c1:

                    st.metric(
                        "Mango",
                        f"Mango {mango['id']}"
                    )


                with c2:

                    st.metric(
                        "Predicted Class",
                        str(
                            mango["class"]
                        )
                    )


                with c3:

                    if (
                        mango["confidence"]
                        is not None
                    ):

                        st.metric(
                            "SVM Confidence",
                            (
                                f"{mango['confidence']:.2f}%"
                            )
                        )

                    else:

                        st.metric(
                            "SVM Confidence",
                            "N/A"
                        )


                with c4:

                    st.metric(
                        "YOLO Confidence",
                        (
                            f"{mango['detection_confidence'] * 100:.2f}%"
                        )
                    )


                predicted_text = (
                    str(
                        mango["class"]
                    )
                    .strip()
                    .lower()
                )


                if predicted_text == "ripe":

                    st.success(
                        f"✅ Mango {mango['id']} "
                        f"is classified as RIPE."
                    )


                elif predicted_text == "unripe":

                    st.warning(
                        f"🟢 Mango {mango['id']} "
                        f"is classified as UNRIPE."
                    )


                elif predicted_text in [
                    "over ripe",
                    "overripe"
                ]:

                    st.error(
                        f"🔴 Mango {mango['id']} "
                        f"is classified as OVER RIPE."
                    )


                elif (
                    predicted_text
                    ==
                    "partially ripe"
                ):

                    st.info(
                        f"🟠 Mango {mango['id']} "
                        f"is classified as "
                        f"PARTIALLY RIPE."
                    )


                else:

                    st.info(
                        f"Mango {mango['id']} "
                        f"Prediction: "
                        f"{mango['class']}"
                    )


                st.divider()


            # Select an individual mango
            st.header(
                "🔬 Image Processing Techniques"
            )


            st.write(
                "Select a detected mango to view "
                "its preprocessing stages."
            )


            mango_options = [

                f"Mango {mango['id']}"

                for mango
                in mango_results
            ]


            selected_mango_label = st.selectbox(
                "Select Mango",
                mango_options,
                key="selected_mango"
            )


            selected_id = int(
                selected_mango_label
                .split()[-1]
            )


            selected_mango = next(

                mango

                for mango
                in mango_results

                if mango["id"]
                ==
                selected_id
            )


            processing = (
                selected_mango[
                    "processing"
                ]
            )


            # Display selected mango ROI
            st.subheader(
                f"Selected: "
                f"{selected_mango_label}"
            )


            st.image(
                bgr_to_rgb(
                    selected_mango[
                        "roi"
                    ]
                ),
                caption=(
                    f"{selected_mango_label} "
                    f"YOLO ROI"
                ),
                use_container_width=True
            )


            # Display resize and illumination correction
            r1c1, r1c2 = st.columns(2)


            with r1c1:

                st.subheader(
                    "1. Resize + Padding"
                )


                st.image(
                    bgr_to_rgb(
                        processing[
                            "original"
                        ]
                    ),
                    caption=(
                        "Aspect-Ratio Preserving "
                        "Resize + Padding "
                        "(224 × 224)"
                    ),
                    use_container_width=True
                )


            with r1c2:

                st.subheader(
                    "2. Illumination Correction"
                )


                st.image(
                    bgr_to_rgb(
                        processing[
                            "corrected"
                        ]
                    ),
                    caption=(
                        "CLAHE on CIE Lab "
                        "L* Channel"
                    ),
                    use_container_width=True
                )


            # Display denoising and segmentation mask
            r2c1, r2c2 = st.columns(2)


            with r2c1:

                st.subheader(
                    "3. Denoising"
                )


                st.image(
                    bgr_to_rgb(
                        processing[
                            "denoised"
                        ]
                    ),
                    caption=(
                        "Bilateral Filter"
                    ),
                    use_container_width=True
                )


            with r2c2:

                st.subheader(
                    "4. Foreground Mask"
                )


                mask_display = (
                    np.asarray(
                        processing[
                            "mask"
                        ],
                        dtype=np.uint8
                    )
                    *
                    255
                )


                st.image(
                    mask_display,
                    caption=(
                        "GrabCut + "
                        "Morphological Refinement"
                    ),
                    clamp=True,
                    use_container_width=True
                )


            # Display segmented mango and Lab colour space
            r3c1, r3c2 = st.columns(2)


            with r3c1:

                st.subheader(
                    "5. Segmented Mango"
                )


                st.image(
                    bgr_to_rgb(
                        processing[
                            "segmented"
                        ]
                    ),
                    caption=(
                        "Foreground Segmentation "
                        "Result"
                    ),
                    use_container_width=True
                )


            with r3c2:

                st.subheader(
                    "6. CIE Lab Colour Space"
                )


                lab_bgr = cv.cvtColor(
                    processing[
                        "lab"
                    ],
                    cv.COLOR_LAB2BGR
                )


                st.image(
                    bgr_to_rgb(
                        lab_bgr
                    ),
                    caption=(
                        "CIE Lab Colour "
                        "Representation"
                    ),
                    use_container_width=True
                )


            # Display LBP texture
            st.subheader(
                "7. LBP Texture Feature"
            )


            lbp = np.asarray(
                processing[
                    "lbp"
                ]
            )


            lbp_display = cv.normalize(
                lbp,
                None,
                0,
                255,
                cv.NORM_MINMAX
            )


            lbp_display = (
                lbp_display.astype(
                    np.uint8
                )
            )


            st.image(
                lbp_display,
                caption=(
                    "Local Binary Pattern "
                    "Texture Representation"
                ),
                clamp=True,
                use_container_width=True
            )

            # Display selected mango classification
            st.divider()


            st.header(
                "🥭 Selected Mango Classification"
            )


            result1, result2, result3, result4 = (
                st.columns(4)
            )


            with result1:

                st.metric(
                    "Mango",
                    selected_mango_label
                )


            with result2:

                st.metric(
                    "Predicted Class",
                    str(
                        selected_mango[
                            "class"
                        ]
                    )
                )


            with result3:

                if (
                    selected_mango[
                        "confidence"
                    ]
                    is not None
                ):

                    st.metric(
                        "SVM Confidence",
                        (
                            f"{selected_mango['confidence']:.2f}%"
                        )
                    )

                else:

                    st.metric(
                        "SVM Confidence",
                        "N/A"
                    )


            with result4:

                st.metric(
                    "YOLO Confidence",
                    (
                        f"{selected_mango['detection_confidence'] * 100:.2f}%"
                    )
                )

            if selected_class == "ripe":

                st.success(
                    "✅ This mango is "
                    "classified as RIPE."
                )


            elif selected_class == "unripe":

                st.warning(
                    "🟢 This mango is "
                    "classified as UNRIPE."
                )


            elif selected_class in [
                "over ripe",
                "overripe"
            ]:

                st.error(
                    "🔴 This mango is "
                    "classified as OVER RIPE."
                )


            elif (
                selected_class
                ==
                "partially ripe"
            ):

                st.info(
                    "🟠 This mango is "
                    "classified as PARTIALLY RIPE."
                )


            else:

                st.info(
                    f"Prediction: "
                    f"{selected_mango['class']}"
                )


            # Reset the application
            st.divider()


            if st.button(
                "🔄 Analyse Another Image",
                key="analyse_another_image"
            ):

                st.session_state.uploader_key += 1

                st.session_state.mango_results = None

                st.session_state.annotated_image = None

                st.session_state.analysed_filename = None


                if (
                    "selected_mango"
                    in st.session_state
                ):

                    del st.session_state[
                        "selected_mango"
                    ]


                st.rerun()