import streamlit as st
import joblib
import cv2 as cv
import numpy as np

from mango_processing import process_mango_image

# PAGE CONFIG
st.set_page_config(
    page_title="Mango Ripeness Detection",
    page_icon="🥭",
    layout="wide"
)

st.title("🥭 Mango Ripeness Detection System")

st.write(
    "Upload a mango image to view each image processing "
    "technique and predict the mango ripeness class."
)

# LOAD MODEL + SCALER
@st.cache_resource
def load_model():

    model = joblib.load(
        "mango_svm_model.pkl"
    )

    scaler = joblib.load(
        "mango_scaler.pkl"
    )

    return model, scaler


model, scaler = load_model()

# Check image quality 
def check_image_quality(image_bgr):

    gray = cv.cvtColor(
        image_bgr,
        cv.COLOR_BGR2GRAY
    )

    # 1. Brightness
    brightness = np.mean(gray)

    if brightness < 60:
        brightness_status = "Too Dark"

    elif brightness > 210:
        brightness_status = "Too Bright"

    else:
        brightness_status = "Good"


    # 2. Sharpness / Blur
    sharpness = cv.Laplacian(
        gray,
        cv.CV_64F
    ).var()

    if sharpness < 50:
        sharpness_status = "Blurry"

    elif sharpness < 100:
        sharpness_status = "Slightly Blurry"

    else:
        sharpness_status = "Good"


    # 3. Resolution
    height, width = image_bgr.shape[:2]

    if width < 224 or height < 224:
        resolution_status = "Low"

    else:
        resolution_status = "Good"


    # 4. Overall quality
    if (
        brightness_status != "Good"
        or sharpness_status == "Blurry"
        or resolution_status == "Low"
    ):
        overall = "Poor"

    else:
        overall = "Acceptable"


    return {
        "brightness": brightness,
        "brightness_status": brightness_status,
        "sharpness": sharpness,
        "sharpness_status": sharpness_status,
        "width": width,
        "height": height,
        "resolution_status": resolution_status,
        "overall": overall
    }

# PREDICTION FUNCTION
def predict_mango(image_bgr):

    # Run your existing preprocessing pipeline
    result = process_mango_image(
        image_bgr
    )

    # Get extracted features
    features = np.array(
        result["features"]
    ).reshape(1, -1)

    # Scale using same scaler from training
    scaled_features = scaler.transform(
        features
    )

    # SVM prediction
    prediction = model.predict(
        scaled_features
    )[0]

    # Confidence
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            scaled_features
        )[0]

        confidence = np.max(
            probabilities
        ) * 100

    else:

        confidence = None

    return result, prediction, confidence

# HELPER FUNCTION FOR BGR -> RGB DISPLAY
def bgr_to_rgb(image):

    return cv.cvtColor(
        image,
        cv.COLOR_BGR2RGB
    )

# IMAGE UPLOAD
uploaded_image = st.file_uploader(
    "Upload Mango Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_image is not None:

    # Convert uploaded file into OpenCV image
    file_bytes = np.asarray(
        bytearray(
            uploaded_image.read()
        ),
        dtype=np.uint8
    )

    image_bgr = cv.imdecode(
        file_bytes,
        cv.IMREAD_COLOR
    )

    quality = check_image_quality(
        image_bgr
    )

    st.subheader(
        "🖼️ Image Quality Assessment"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Brightness",
            quality["brightness_status"]
        )

    with col2:
        st.metric(
            "Sharpness",
            quality["sharpness_status"]
        )

    with col3:
        st.metric(
            "Resolution",
            quality["resolution_status"]
        )


    # Show warning
    if quality["overall"] == "Poor":

        st.warning(
            "⚠️ Poor image quality detected. "
            "The image may affect prediction accuracy."
        )

    else:

        st.success(
            "✅ Image quality is acceptable for analysis."
        )


    if image_bgr is None:

        st.error(
            "Unable to read the uploaded image."
        )

        st.stop()


    # SHOW UPLOADED IMAGE
    st.subheader(
        "📷 Uploaded Image"
    )

    st.image(
        bgr_to_rgb(image_bgr),
        caption="Input Mango Image",
        use_container_width=True
    )

    # ANALYSE BUTTON
    if st.button(
        "Analyse Mango",
        type="primary"
    ):

        try:

            result, prediction, confidence = (
                predict_mango(
                    image_bgr
                )
            )

            # IMAGE PROCESSING PIPELINE
            st.divider()

            st.header(
                "🔬 Image Processing Techniques"
            )


            # ROW 1
            # ORIGINAL + MSR
            col1, col2 = st.columns(2)

            with col1:

                st.subheader(
                    "1. Original / Resized Image"
                )

                st.image(
                    bgr_to_rgb(
                        result["original"]
                    ),
                    caption="Original Image (224 × 224)",
                    use_container_width=True
                )


            with col2:

                st.subheader(
                    "2. Multi-Scale Retinex"
                )

                st.image(
                    bgr_to_rgb(
                        result["msr"]
                    ),
                    caption="Illumination and Shadow Correction",
                    use_container_width=True
                )


            # ROW 2
            # CLAGC + DENOISING
            col3, col4 = st.columns(2)

            with col3:

                st.subheader(
                    "3. CLAGC Enhancement"
                )

                st.image(
                    bgr_to_rgb(
                        result["clagc"]
                    ),
                    caption="Gamma Correction + CLAHE",
                    use_container_width=True
                )


            with col4:

                st.subheader(
                    "4. Denoising"
                )

                st.image(
                    bgr_to_rgb(
                        result["denoised"]
                    ),
                    caption="Bilateral Filter",
                    use_container_width=True
                )

            # ROW 3
            # MASK + SEGMENTATION
            col5, col6 = st.columns(2)

            with col5:

                st.subheader(
                    "5. Foreground Mask"
                )

                mask_display = (
                    result["mask"] * 255
                ).astype(
                    np.uint8
                )

                st.image(
                    mask_display,
                    caption="Otsu Threshold + Morphological Cleanup",
                    clamp=True,
                    use_container_width=True
                )


            with col6:

                st.subheader(
                    "6. Segmented Mango"
                )

                st.image(
                    bgr_to_rgb(
                        result["segmented"]
                    ),
                    caption="Foreground Segmentation Result",
                    use_container_width=True
                )

            # ROW 4
            # LAB + LBP

            col7, col8 = st.columns(2)

            with col7:

                st.subheader(
                    "7. CIE Lab Colour Space"
                )

                # Lab should not be shown directly as RGB.
                # Convert Lab back into RGB for visual display.
                lab_bgr = cv.cvtColor(
                    result["lab"],
                    cv.COLOR_LAB2BGR
                )

                st.image(
                    bgr_to_rgb(
                        lab_bgr
                    ),
                    caption="CIE Lab Representation",
                    use_container_width=True
                )


            with col8:

                st.subheader(
                    "8. LBP Texture"
                )

                lbp = result["lbp"]

                # Normalize LBP for visual display
                lbp_display = cv.normalize(
                    lbp,
                    None,
                    0,
                    255,
                    cv.NORM_MINMAX
                )

                lbp_display = (
                    lbp_display
                ).astype(
                    np.uint8
                )

                st.image(
                    lbp_display,
                    caption="Local Binary Pattern Texture",
                    clamp=True,
                    use_container_width=True
                )

            # FINAL CLASSIFICATION

            st.divider()

            st.header(
                "🥭 Mango Ripeness Classification"
            )

            if confidence is not None:

                col9, col10 = st.columns(2)

                with col9:

                    st.metric(
                        "Predicted Mango Class",
                        str(prediction)
                    )

                with col10:

                    st.metric(
                        "Confidence",
                        f"{confidence:.2f}%"
                    )

            else:

                st.metric(
                    "Predicted Mango Class",
                    str(prediction)
                )


            # RESULT MESSAGE

            predicted_text = str(
                prediction
            ).lower()

            if predicted_text == "ripe":

                st.success(
                    "✅ The mango is classified as RIPE."
                )

            elif predicted_text == "unripe":

                st.warning(
                    "🟡 The mango is classified as UNRIPE."
                )

            elif predicted_text == "overripe":

                st.error(
                    "🔴 The mango is classified as OVERRIPE."
                )

            else:

                st.info(
                    f"Prediction: {prediction}"
                )


        except Exception as e:

            st.error(
                f"Image processing error: {e}"
            )