# Import libraries for data analysis and visualization
import cv2
import numpy as np
import os
import streamlit as st
import tempfile
import tensorflow as tf
import torch
import torch.nn as nn
from io import BytesIO
from PIL import Image

#--------------Load models and images----------------------
# Load default image
default_img = "Brain_Tumor_Segmentation_Dataset/image/0/Tr-no_0011.jpg"
# Load default image mask
default_mask = "Brain_Tumor_Segmentation_Dataset/mask/0/Tr-no_0011_m.jpg"

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_classification_model():
    # Load pre-trained CNN model
    return tf.keras.models.load_model('model_classification_original_7000_dataset.keras')
#Load U-Net architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)
        self.decoder1 = self.conv_block(1024 + 512, 512)
        self.decoder2 = self.conv_block(512 + 256, 256)
        self.decoder3 = self.conv_block(256 + 128, 128)
        self.decoder4 = self.conv_block(128 + 64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        dec1 = self.decoder1(torch.cat([nn.functional.interpolate(bottleneck, scale_factor=2), enc4], dim=1))
        dec2 = self.decoder2(torch.cat([nn.functional.interpolate(dec1, scale_factor=2), enc3], dim=1))
        dec3 = self.decoder3(torch.cat([nn.functional.interpolate(dec2, scale_factor=2), enc2], dim=1))
        dec4 = self.decoder4(torch.cat([nn.functional.interpolate(dec3, scale_factor=2), enc1], dim=1))
        output = self.final_conv(dec4)
        return output

@st.cache_resource
def load_segmentation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    #model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.load_state_dict(torch.load("best_model_ UNet_5_focal_ESTOP_64batch.pt", map_location=device))

    model.eval()
    return model, device

def preprocess_image_classification(image):
    img = cv2.resize(image, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return img

def preprocess_image_segmentation(image, device):
    img = Image.open(image).convert('L')
    img = img.resize((256, 256))
    img = np.array(img) / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    return img

def classify_image(model, image):
    processed_img = preprocess_image_classification(image)
    processed_img = np.expand_dims(processed_img, axis=0)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    return class_labels[predicted_class], predictions[0][predicted_class]

def segment_image(_model, _image, _device):
    processed_img = preprocess_image_segmentation(_image, _device)
    with torch.no_grad():
        output_mask = _model(processed_img)
        predicted_mask = (torch.sigmoid(output_mask) > 0.5).float()
    return np.array(predicted_mask[0][0].cpu())

# Load models
classification_model = load_classification_model()
segmentation_model, device = load_segmentation_model()

#st.title("Model Demonstration")
st.title("Brain Tumor Analysis Application")
#st.title("Brain Tumor Detection & Demonstration")


st.sidebar.image("brain-tumor-classification-and-segmentation-high-resolution-logo.png", width=300) # Imagepath

st.markdown(
    """
    <style>
        .brand-title {
            font-family: 'Arial Black', sans-serif;
            color: #4CAF50;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.5);
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <div class='brand-title' style='text-align:center;'>
        <h1>Classify & Segment</h1>
        <h2>Brain MRI Images</h2>
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("Upload MRI Images"):
    uploaded_img = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="uploaded_img", help="Upload a JPG, JPEG, or PNG image")

with st.expander("Upload Ground Truth-Optional"):
    uploaded_ground_truth = st.file_uploader("", type=["jpg", "jpeg", "png"], accept_multiple_files=False, key="uploaded_another_image", help="Upload a JPG, JPEG, or PNG image")

col1, col2, col3, col4 = st.columns(4)

#-----------------------------------------------------

zooming = False
if uploaded_img is not None:
    img_bytes = uploaded_img.read()
    if img_bytes:
        try:
            img_array = np.frombuffer(img_bytes, np.uint8)
            mri_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            scaled_mri_image = cv2.resize(mri_image, (256, 256))
            col2.image(scaled_mri_image, caption="", use_column_width=True, output_format="JPEG")
            
            col2_1, col2_2 = col2.columns([1, 3])
            col2_2.markdown("<p style='text-align: left;'><strong>MRI Image</strong></p>", unsafe_allow_html=True)
            zooming = col2_1.checkbox("")
            if zooming:
                scale_coefficient = 1.375
                scaled_width = int(mri_image.shape[1] * scale_coefficient)
                scaled_height = int(mri_image.shape[0] * scale_coefficient)
                scaled_image = cv2.resize(mri_image, (scaled_width, scaled_height))
                
                
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.error("Empty image data. Please upload a valid image file.")

if uploaded_ground_truth is not None:
    #ground_truth = Image.open(uploaded_another_image)
    img_bytes_gt = uploaded_ground_truth.read()
    img_array_gt = np.frombuffer(img_bytes_gt, np.uint8)
    ground_truth = cv2.imdecode(img_array_gt, cv2.IMREAD_COLOR)
    scaled_ground_truth = cv2.resize(ground_truth, (256, 256))
    col3.image(scaled_ground_truth, caption="", use_column_width=True, output_format="JPEG")
    col3.markdown("<p style='text-align: center;'><strong>Ground Truth</strong></p>", unsafe_allow_html=True)
#-----------------------------------------------------

if uploaded_img is not None:
    with col1:
        with st.expander("Start &nbsp;&nbsp; Classification"):
            if st.button("Run Classification Model"):
                with st.spinner(text="Classification in progress..."):
                    tumor_type, confidence = classify_image(classification_model, mri_image)
                    st.session_state.classifier_output = {'class': tumor_type, 'confidence': confidence}
                    st.success('Done!')

#-----------------------------------------------------------------------------------------------------------------------------

container = st.empty()

with container.container():
    st.markdown(
        """
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px; font-family: Times New Roman, serif; color: blue; height: 150px'>
            <h3 style='color: black;'> </h3> 
            <p><span style='color: black;'><strong> </strong></span> </p>
            <p><span style='color: black;'><strong> </strong></span> </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Check if classifier output exists
if 'classifier_output' in st.session_state:
    with container.container():
        st.markdown(
            """
            <div style='background-color: #f0f0f0; padding: 10px; border-radius: 10px;  color: blue; height: 150px;'>
                <h3 style='color: black;'>Classification Results</h3>
                <p><span style='color: black;'><strong>Tumor type:</strong></span> {}</p>
                <p><span style='color: black;'><strong>Confidence:</strong></span> {:.2f}%</p>
            </div>
            """.format(
                st.session_state.classifier_output["class"],
                st.session_state.classifier_output["confidence"] * 100
            ),
            unsafe_allow_html=True
        )   #font-family: Times New Roman, serif; font-family: Times New Roman, serif;
#-----------------------------------------------------
segmentation_run = False
zooming_segmentation=False

if uploaded_img is not None:
    with col1:
        with st.expander("Start &nbsp;&nbsp; Segmentation"):
            if st.button("Run Segmentation Model"):
                with st.spinner(text="Segmentation in progress..."):
                    predicted_mask = segment_image(segmentation_model, uploaded_img, device)
                    st.session_state.segmentation_output = {'predicted_mask': predicted_mask}
                    st.session_state.segmentation_run = True
                    st.success('Segmentation Done!')
#
# Display segmentation result
if 'segmentation_run' in st.session_state and st.session_state.segmentation_run:
    scaled_predicted_mask = cv2.resize(st.session_state.segmentation_output["predicted_mask"], (256, 256))
    col4.image(scaled_predicted_mask, caption="", use_column_width=True, clamp=True, output_format="JPEG")
    # Can not use the caption argument to set the text to bold. Need to use Markdown as given below
    col4_1, col4_2 = col4.columns([1,3])
    # Bold caption text using Markdown functionality
    col4_2.markdown("<p style='text-align: left;'><strong>Segmentation</strong></p>", unsafe_allow_html=True)
    # Add col4_1, col4_2 and the checkbox here
    zooming_segmentation = col4_1.checkbox(" ")# code was giving errors stating that there are to similar widgets and had to add just a 'space' here to fix it. 
    # Otherwise, display the segmentation mask without zooming    
    # Bold caption text using Markdown functionality
    # Otherwise, display the segmentation mask without zooming
    if zooming_segmentation:
        # If the checkbox for zooming segmentation is checked, apply zooming

        scale_coefficient = 1.375
        scaled_width = int(mri_image.shape[1] * scale_coefficient)
        scaled_height = int(mri_image.shape[0] * scale_coefficient)
        scaled_mask = cv2.resize(st.session_state.segmentation_output["predicted_mask"], (scaled_width, scaled_height))
        

#------------------------------------------------------

# Placeholder for segmentation_output (replace with actual output)
if uploaded_img is not None:
    with col1.expander("Save"):
        if 'segmentation_output' in st.session_state:
            # Get the predicted mask
            predicted_mask = st.session_state.segmentation_output["predicted_mask"]

            # Normalize the mask to have values in range 0-255 again before exporting
            normalized_mask = (predicted_mask * 255).astype(np.uint8)

            # Convert to PIL Image and save
            predicted_mask_pil = Image.fromarray(normalized_mask)

            # Save the predicted mask as a PNG image
            with tempfile.TemporaryDirectory() as temp_dir:
                save_filename = "predicted_mask.png"
                save_fullpath = os.path.join(temp_dir, save_filename)
                predicted_mask_pil.save(save_fullpath)

                # Display download button
                with open(save_fullpath, "rb") as file:
                    if st.download_button("Download Predicted Mask", file.read(), file_name=save_filename):
                        st.success("Saved successfully!")
        else:
            st.markdown("<p style='color: red; text-align: center;'><strong>Run Segmentation Model before!</strong></p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color: red; text-align: center;'><strong>No MRI image uploaded!</strong></p>", unsafe_allow_html=True)


#----------------------------------------------------
# Display the scaled image in the main page
if zooming:

    st.image(scaled_image, caption="", use_column_width=False, clamp=True, output_format="JPEG")    
    st.markdown("<p style='text-align: center;'><strong>Zooming MRI Image</strong></p>", unsafe_allow_html=True)
if zooming==False:
    pass
if zooming_segmentation:
    st.image(scaled_mask, caption="", use_column_width=True, clamp=True, output_format="JPEG")
    st.markdown("<p style='text-align: center;'><strong>Zooming Segmentation</strong></p>", unsafe_allow_html=True)
if zooming_segmentation==False:
    pass


#-----------------------------------------------------


# Function to display help information for each task
def display_help(task):
    help_info = {
        "Task 1": "Description of task 1.",
        "Task 2": "Description of task 2.",
        "Task 3": "Description of task 3."
    }
    st.markdown(f"## Help Information for {task}")
    st.markdown(help_info[task])
    st.markdown("For more information, visit our [documentation](https://example.com).")

# Add empty lines to push the Help expander to the bottom
for _ in range(7):
    st.sidebar.write("")

side_1, side_2 = st.sidebar.columns([2, 1.4])

with side_1:
    # Sidebar expander for help information
    with st.expander("Help"):
        if st.button('General'):
            display_help("Task 1")
        if st.button('Classification'):
            display_help("Task 2")
        if st.button('Segmentation'):
            display_help("Task 3")
        if st.button('NeuroVision'):
            display_help("Task 1")
        if st.button('Support'):
            display_help("Task 3")
