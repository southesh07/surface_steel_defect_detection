import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
import io
import cv2

# ---- Setup the page ----
st.set_page_config(page_title="Scratch Detection", page_icon="🔍", layout="centered")

# ---- Load the AI model (only loads once, then reuses) ----
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---- Function: Make the image clearer using CLAHE ----
def enhance_image(image):
    # Convert image to numpy array (numbers)
    img_array = np.array(image.convert("RGB"))
    # Convert to LAB color format so we can boost brightness only
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    # Apply CLAHE to the brightness channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_enhanced = cv2.merge([clahe.apply(l), a, b])
    # Convert back to RGB and return as a PIL image
    return Image.fromarray(cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB))

# ---- Function: Download an image from a URL ----
def download_image(url):
    try:
        response = requests.get(url.strip(), headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
        if "image" not in response.headers.get("Content-Type", ""):
            return None, "That URL does not point to an image."
        return Image.open(io.BytesIO(response.content)).convert("RGB"), None
    except requests.exceptions.MissingSchema:
        return None, "Invalid URL — it must start with http:// or https://"
    except requests.exceptions.Timeout:
        return None, "The request timed out. Try uploading the image instead."
    except Exception as e:
        return None, f"Something went wrong: {e}"

# ---- Page title ----
st.title("🔍 Scratch Detection")
st.caption("Upload a photo or paste an image link to check for scratches.")
st.divider()

# These will hold the image and whether the detect button was clicked
image = None
detect_clicked = False

# ---- Two tabs: one for uploading, one for a URL ----
tab_upload, tab_url = st.tabs(["📁 Upload Image", "🌐 Image URL"])

# ---- Tab 1: Upload a photo from your computer ----
with tab_upload:
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Open the uploaded file as an image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        st.divider()

        # Settings
        conf_threshold = st.slider(
            "Sensitivity (Confidence Threshold)",
            min_value=0.05, max_value=0.10, value=0.10, step=0.05,
            help="Lower = finds more scratches. Higher = only shows obvious ones."
        )
        use_enhance = st.toggle("Boost Image Contrast", value=True,
                                help="Helps find faint scratches by improving image clarity.")
        detect_clicked = st.button("⚡ Detect Scratches", use_container_width=True, type="primary")

# ---- Tab 2: Paste an image URL ----
with tab_url:
    url_input = st.text_input("Paste an image URL here", placeholder="https://example.com/photo.jpg")

    if st.button("Load Image from URL", use_container_width=True):
        if url_input:
            with st.spinner("Downloading image..."):
                downloaded_image, error = download_image(url_input)
            if error:
                st.error(error)
            else:
                # Save it so it stays after the button is clicked
                st.session_state["url_image"] = downloaded_image
                st.success("Image loaded successfully!")
        else:
            st.warning("Please paste a URL first.")

    if "url_image" in st.session_state:
        image = st.session_state["url_image"]
        st.image(image, use_container_width=True)
        st.divider()

        conf_threshold = st.slider(
            "Sensitivity (Confidence Threshold)",
            min_value=0.05, max_value=0.10, value=0.10, step=0.05,
            help="Lower = finds more scratches. Higher = only shows obvious ones.",
            key="url_conf"
        )
        use_enhance = st.toggle("Boost Image Contrast", value=True,
                                help="Helps find faint scratches by improving image clarity.",
                                key="url_enhance")
        detect_clicked = st.button("⚡ Detect Scratches", use_container_width=True,
                                   type="primary", key="url_detect")

# ---- Run detection when the button is clicked ----
if detect_clicked and image is not None:

    with st.spinner("Analyzing image for scratches..."):
        # Step 1: Optionally enhance the image
        processed_image = enhance_image(image) if use_enhance else image

        # Step 2: Run the AI model
        results = model.predict(source=processed_image, imgsz=640, conf=conf_threshold)

    # Count how many scratches were found
    boxes = results[0].boxes
    num_scratches = len(boxes) if boxes is not None else 0

    # Show the annotated image
    st.divider()
    st.subheader("Detection Result")
    st.image(results[0].plot(), channels="BGR", use_container_width=True)
    st.divider()

    # Calculate confidence stats
    if num_scratches > 0:
        confidences = [float(b.conf[0]) for b in boxes]
        avg_conf = float(np.mean(confidences))
        max_conf = float(np.max(confidences))
    else:
        avg_conf = 0.0
        max_conf = 0.0

    # Show summary numbers
    col1, col2, col3 = st.columns(3)
    col1.metric("Scratches Found", num_scratches)
    col2.metric("Avg Confidence", f"{avg_conf:.1%}")
    col3.metric("Max Confidence", f"{max_conf:.1%}")
    st.divider()

    # Show details for each scratch
    if num_scratches > 0:
        st.subheader("Detected Scratches")
        for box in boxes:
            label = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            st.progress(confidence, text=f"**{label}** — {confidence:.2%} confident")
    else:
        st.success("✅ No scratches detected — the surface looks clean!")

# ---- Footer ----
st.divider()
st.caption("SteelGuard AI · Powered by YOLOv11 · Built with Streamlit")