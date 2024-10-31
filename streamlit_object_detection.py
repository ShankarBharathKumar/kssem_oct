import streamlit as st
import os
import subprocess
from PIL import Image
from pathlib import Path

# Paths to your model and the folder where YOLOv5 stores outputs
WEIGHTS_PATH = '/home/bharath/avua_video_resume/2254/kssem/github/kssem_oct/trained_model/best.pt'
DETECT_SCRIPT = '/home/bharath/avua_video_resume/2254/kssem/github/kssem_oct/yolov5/detect.py'
OUTPUT_DIR = '/home/bharath/avua_video_resume/2254/kssem/github/kssem_oct/yolov5/runs/detect'

# Set page configuration to wide layout
st.set_page_config(layout="wide")

# Center the title using HTML and CSS
st.markdown(
    """
    <h1 style='text-align: center;'>Objects Detection using Computer Vision</h1>
    """,
    unsafe_allow_html=True
)
st.write("Upload an image and see the model's prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image on the left
    col1, col2 = st.columns(2)
    input_image = Image.open(uploaded_file)
    col1.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Save uploaded file temporarily
    input_image_path = "/tmp/uploaded_image.png"
    with open(input_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run the YOLOv5 detection command
    command = [
        "python", DETECT_SCRIPT,
        "--weights", WEIGHTS_PATH,
        "--img", "640",
        "--conf", "0.25",
        "--source", input_image_path
    ]
    with st.spinner("Processing..."):
        subprocess.run(command, check=True)

    # Find the latest output folder
    output_folders = sorted(Path(OUTPUT_DIR).glob('exp*'), key=os.path.getmtime, reverse=True)
    latest_folder = output_folders[0] if output_folders else None

    if latest_folder:
        # Get the result image from the latest folder
        result_images = list(latest_folder.glob("*.jpg")) + list(latest_folder.glob("*.png"))
        if result_images:
            result_image_path = result_images[0]
            result_image = Image.open(result_image_path)

            # Display the result image in the right column
            col2.image(result_image, caption="Detected Image", use_column_width=True)
        else:
            st.write("No result image found.")
    else:
        st.write("No output folder found.")
